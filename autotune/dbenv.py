import os
import time
import math
import threading
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Manager
from collections import defaultdict
from typing import Any
from .knobs import gen_continuous
import pdb
import sys
import math
from .knobs import logger
from .utils.parser import ConfigParser
from .utils.parser import parse_tpcc, parse_sysbench, parse_oltpbench, parse_cloudbench, parse_job
from .knobs import initialize_knobs, save_knobs, get_default_knobs, knob2action
from dynaconf import settings
import re
import psutil
import multiprocessing as mp
from .resource_monitor import ResourceMonitor

im_alive = mp.Value('b', False)
CPU_CORE = 8
TIMEOUT=180
TIMEOUT_CLOSE=90
RESTART_FREQUENCY = 20

value_type_metrics = [
    'lock_deadlocks', 'lock_timeouts', 'lock_row_lock_time_max',
    'lock_row_lock_time_avg', 'buffer_pool_size', 'buffer_pool_pages_total',
    'buffer_pool_pages_misc', 'buffer_pool_pages_data', 'buffer_pool_bytes_data',
    'buffer_pool_pages_dirty', 'buffer_pool_bytes_dirty', 'buffer_pool_pages_free',
    'trx_rseg_history_len', 'file_num_open_files', 'innodb_page_size']

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
log_num_default = 2
log_size_default = 50331648

def generate_knobs(action, method):
    if method in ['ddpg', 'ppo', 'sac', 'gp']:
        return gen_continuous(action)
    else:
        raise NotImplementedError("Not implemented generate_knobs")



class DBEnv(ABC):
    def __init__(self, workload):
        self.score = 0.
        self.steps = 0
        self.terminate = False
        self.workload = workload

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self, knobs, episode, step):
        pass

    @abstractmethod
    def terminate(self):
        return False


class MySQLEnv(DBEnv):
    def __init__(self,
                 workload,
                 knobs_config,
                 num_metrics,
                 log_path='',
                 threads=8,
                 host='localhost',
                 port=3392,
                 user='root',
                 passwd='',
                 dbname='tpcc',
                 sock='',
                 rds_mode=False,
                 workload_zoo_config='',
                 workload_zoo_app='',
                 oltpbench_config_xml='',
                 disk_name='nvme1n1',
                 tps_constraint=0,
                 latency_constraint=0,
                 pid=9999,
                 knob_num=-1,
                 y_variable='tps',
                 lhs_log='output.res'
                 ):
        super().__init__(workload)
        self.knobs_config = knobs_config
        self.mysqld = os.environ.get('MYSQLD')
        self.mycnf = os.environ.get('MYCNF')
        if not self.mysqld:
            logger.error('You should set MYSQLD env var before running the code.')
        if not self.mycnf:
            logger.error('You should set MYCNF env var before running the code.')
        self.workload = workload
        self.log_path = log_path
        self.num_metrics = num_metrics
        self.external_metricsdefault_ = []
        self.last_external_metrics = []
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.sock = sock
        self.threads = threads
        self.best_result = './autotune_best.res'
        self.knobs_config = knobs_config
        self.knobs_detail = initialize_knobs(knobs_config, knob_num)
        self.default_knobs = get_default_knobs()
        self.rds_mode = rds_mode
        self.oltpbench_config_xml = oltpbench_config_xml
        self.step_count = 0
        self.disk_name = disk_name
        self.workload_zoo_config = workload_zoo_config
        self.workload_zoo_app = workload_zoo_app
        self.tps_constraint = tps_constraint
        self.latency_constraint = latency_constraint
        self.pre_combine_log_file_size = 0
        self.connect_sucess = True
        self.pid = pid
        self.reinit_interval = 0
        self.reinit = True
        if self.rds_mode:
            self.reinit = False
        self.generate_time()
        self.y_variable = y_variable
        self.lhs_log = lhs_log

    def generate_time(self):
        global BENCHMARK_RUNNING_TIME
        global BENCHMARK_WARMING_TIME
        global TIMEOUT
        global RESTART_FREQUENCY

        if self.workload['name'] == 'sysbench' or self.workload['name'] == 'oltpbench':
            BENCHMARK_RUNNING_TIME = 120
            BENCHMARK_WARMING_TIME = 30
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME + 15
            RESTART_FREQUENCY = 200
        if self.workload['name'] == 'job':
            BENCHMARK_RUNNING_TIME = 240
            BENCHMARK_WARMING_TIME = 0
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME
            RESTART_FREQUENCY = 30000

    def apply_knobs(self, knobs):
        self._kill_mysqld()
        modify_concurrency = False
        if 'innodb_thread_concurrency' in knobs.keys() and knobs['innodb_thread_concurrency'] * (200 * 1024) > self.pre_combine_log_file_size:
            true_concurrency = knobs['innodb_thread_concurrency']
            modify_concurrency = True
            knobs['innodb_thread_concurrency'] = int(self.pre_combine_log_file_size / (200 * 1024.0)) - 2
            logger.info("modify innodb_thread_concurrency")

        if 'innodb_log_file_size' in knobs.keys():
            log_size = knobs['innodb_log_file_size']
        else:
            log_size = log_size_default
        if 'innodb_log_files_in_group' in knobs.keys():
            log_num =  knobs['innodb_log_files_in_group']
        else:
            log_num = log_num_default

        if 'innodb_thread_concurrency' in knobs.keys() and knobs['innodb_thread_concurrency'] * (200 * 1024) > log_num * log_size:
            logger.info("innodb_thread_concurrency is set too large")
            return False

        knobs_rdsL =  self._gen_config_file(knobs)
        sucess = self._start_mysqld()
        try:
            logger.info('sleeping for {} seconds after restarting mysql'.format(RESTART_WAIT_TIME))
            time.sleep(RESTART_WAIT_TIME)

            db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
            sql1 = 'SHOW VARIABLES LIKE "innodb_log_file_size";'
            sql2 = 'SHOW VARIABLES LIKE "innodb_log_files_in_group";'
            r1 = db_conn.fetch_results(sql1)
            file_size = r1[0]['Value'].strip()
            r2 = db_conn.fetch_results(sql2)
            file_num = r2[0]['Value'].strip()
            self.pre_combine_log_file_size = int(file_num) * int(file_size)
            if len(knobs_rdsL) > 0:
                tmp_rds = {}
                for knob_rds in knobs_rdsL:
                    tmp_rds[knob_rds] = knobs[knob_rds]
                self.apply_rds_knobs(tmp_rds)
            if modify_concurrency:
                tmp = {}
                tmp['innodb_thread_concurrency'] = true_concurrency
                self.apply_rds_knobs(tmp)
                knobs['innodb_thread_concurrency'] = true_concurrency
        except:
            sucess = False

        return sucess

    def apply_rds_knobs(self, knobs):
        # self.restart_rds()
        # apply knobs remotely
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        if 'innodb_io_capacity' in knobs.keys():
            self.set_rds_param(db_conn, 'innodb_io_capacity_max', 2 * int(knobs['innodb_io_capacity']))
        # for k, v in knobs.items():
        #   self.set_rds_param(db_conn, k, v)
        for key in knobs.keys():
            self.set_rds_param(db_conn, key, knobs[key])
        db_conn.close_db()

        return True

    def _check_apply(self, db_conn, k, v, v0, IsSession=False):
        if IsSession:
            sql = 'SHOW VARIABLES LIKE "{}";'.format(k)
            r = db_conn.fetch_results(sql)
            if r[0]['Value'] == 'ON':
                vv = 1
            elif r[0]['Value'] == 'OFF':
                vv = 0
            else:
                vv = r[0]['Value'].strip()
            if vv == v0:
                return False
            return True

        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if r[0]['Value'] == 'ON':
            vv = 1
        elif r[0]['Value'] == 'OFF':
            vv = 0
        else:
            vv = r[0]['Value'].strip()
        if vv == v0:
            return False
        return True

    def set_rds_param(self, db_conn, k, v):
        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0
        if r[0]['Value'] == 'ON':
            v0 = 1
        elif r[0]['Value'] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0]['Value'])
            except:
                v0 = r[0]['Value'].strip()
        if v0 == v:
            return True

        IsSession = False
        if str(v).isdigit():
            sql = "SET GLOBAL {}={}".format(k, v)
        else:
            sql = "SET GLOBAL {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {}".format(sql))
            IsSession = True
            if str(v).isdigit():
                sql = "SET {}={}".format(k, v)
            else:
                sql = "SET {}='{}'".format(k, v)
            db_conn.execute(sql)
        while not self._check_apply(db_conn, k, v, v0, IsSession):
            time.sleep(1)
        return True

    def get_external_metrics(self, filename=''):
        """Get the external metrics including tps and rt"""
        result = ''
        if self.workload['name'] == 'tpcc':
            result = parse_tpcc(filename)
        elif self.workload['name'] == 'tpcc_rds':
            result = parse_tpcc(filename)
        elif self.workload['name'] == 'sysbench':
            result = parse_sysbench(filename)
        elif self.workload['name'] == 'sysbench_rds':
            result = parse_sysbench(filename)
        elif self.workload['name'] == 'oltpbench':
            result = parse_oltpbench('results/{}.summary'.format(filename))
        elif self.workload['name'] == 'job':
            dirname, _ = os.path.split(os.path.abspath(__file__))
            select_file = dirname + '/cli/selectedList.txt'
            result = parse_job(filename, select_file)
        else:
            # logger.error('unsupported workload {}'.format(self.workload['name']))
            result = parse_cloudbench(filename)
        return result

    def reset_internal_state(self):
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
        res = db_conn.fetch_results(sql, json=False)
        for (k, _) in res:
            if k in BLACKLIST or k in IMMUTABLELIST:
                continue
            sql1 = 'SET GLOBAL innodb_monitor_disable={}'.format(k)
            db_conn.execute(sql1)
            sql1 = 'SET GLOBAL innodb_monitor_reset_all={}'.format(k)
            db_conn.execute(sql1)
            sql1 = 'SET GLOBAL innodb_monitor_enable={}'.format(k)
            db_conn.execute(sql1)
        time.sleep(5)
        #        res = db_conn.fetch_results(sql, json=False)
        #        for (k, v) in res:
        #            if k in BLACKLIST or k in IMMUTABLELIST:
        #                continue
        #            if v != 0:
        #                logger.error('reset internal state failed. {}:{}'.format(k, v))
        db_conn.close_db()

    def get_db_size(self):
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(
            self.dbname)
        res = db_conn.fetch_results(sql, json=False)
        db_size = float(res[0][0][:-2])
        db_conn.close_db()
        return db_size

    def get_internal_metrics(self, internal_metrics):
        """Get the all internal metrics of MySQL, like io_read, physical_read.

        This func uses a SQL statement to lookup system table: information_schema.INNODB_METRICS
        and returns the lookup result.
        """
        self.connect_sucess = True
        _counter = 0
        _period = 5
        count = (BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME) / _period - 1
        warmup = BENCHMARK_WARMING_TIME / _period

        # self.reset_internal_state()
        # double reset
        # TODO: fix this
        # self.reset_internal_state()

        def collect_metric(counter):
            counter += 1
            print(counter)
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                timer.cancel()
            try:
                db_conn = MysqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
            except:
                if counter > warmup:
                    self.connect_sucess = False
                    logger.info("connection failed during internal metrics collection")
                    return

            try:
                if counter > warmup:

                    sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
                    res = db_conn.fetch_results(sql, json=False)
                    res_dict = {}
                    for (k, v) in res:
                        # if not k in BLACKLIST:
                        res_dict[k] = v
                    internal_metrics.append(res_dict)

            except Exception as err:
                self.connect_sucess = False
                logger.info("connection failed during internal metrics collection")
                logger.info(err)

        collect_metric(_counter)
        # f.close()
        return internal_metrics


    def get_reward(self, external_metrics, y_variable):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            return _reward

        if sum(external_metrics) == 0 or self.default_external_metrics[0] == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # tps
        if y_variable == 'tps':
            delta_0_tps = float((external_metrics[0] - self.default_external_metrics[0])) / self.default_external_metrics[0]
            delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0])) / self.last_external_metrics[0]
            reward = calculate_reward(delta_0_tps, delta_t_tps)

        # latency
        else:
            delta_0_lat = float((-external_metrics[1] + self.default_external_metrics[1])) / self.default_external_metrics[
            1]
            delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]
            reward = calculate_reward(delta_0_lat, delta_t_lat)

        #reward = tps_reward * 0.95 + 0.05 * lat_reward
        #   reward = tps_reward * 0.6 + 0.4 * lat_reward
        self.score += reward

        # if reward > 0:
        #    reward = reward*1000000
        return reward

    def get_reward2(self, external_metrics):
        return float(external_metrics[0] / self.last_external_metrics[0])

    def get_reward3(self, external_metrics):
        return float(external_metrics[0] / self.default_external_metrics[0])

    def _post_handle(self, metrics):
        result = np.zeros(65)

        def do(metric_name, metric_values):
            metric_type = 'counter'
            if metric_name in value_type_metrics:
                metric_type = 'value'
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0]) * 23 / len(metric_values)
            else:
                return float(sum(metric_values)) / len(metric_values)

        keys = list(metrics[0].keys())
        keys.sort()
        total_pages = 0
        dirty_pages = 0
        request = 0
        reads = 0
        page_data = 0
        page_size = 0
        page_misc = 0
        for idx in range(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
            if key == 'buffer_pool_pages_total':
                total_pages = result[idx]
            elif key == 'buffer_pool_pages_dirty':
                dirty_pages = result[idx]
            elif key == 'buffer_pool_read_requests':
                request = result[idx]
            elif key == 'buffer_pool_reads':
                reads = result[idx]
            elif key == 'buffer_pool_pages_data':
                page_data = result[idx]
            elif key == 'innodb_page_size':
                page_size = result[idx]
            elif key == 'buffer_pool_pages_misc':
                page_misc = result[idx]
        dirty_pages_per = dirty_pages / total_pages
        hit_ratio = request / float(request + reads)
        page_data = (page_data + page_misc) * page_size / (1024.0 * 1024.0 * 1024.0)

        return result, dirty_pages_per, hit_ratio, page_data

    def _get_best_now(self):
        with open(self.best_result) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    def _record_best(self, external_metrics):
        best_flag = False
        if os.path.exists(self.best_result):
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) != 0:
                rate = float(tps_best) / lat_best
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                rate_best_now = float(best_now[0]) / float(best_now[1])
                if rate > rate_best_now:
                    best_flag = True
                    with open(self.best_result, 'w') as f:
                        f.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        else:
            file = open(self.best_result, 'w')
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) == 0:
                rate = 0
            else:
                rate = float(tps_best) / lat_best
            file.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        return best_flag

    def _gen_config_file(self, knobs):
        cnf_parser = ConfigParser(self.mycnf)
        # for k, v in knobs.items():
        #    cnf_parser.set(k, v)
        konbs_not_in_mycnf = []
        for key in knobs.keys():
            if not key in self.knobs_detail.keys():
                konbs_not_in_mycnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])
        cnf_parser.replace()
        logger.info('generated config file done')
        return konbs_not_in_mycnf

    def get_benchmark_cmd(self):
        timestamp = int(time.time())
        filename = self.log_path + '/{}.log'.format(timestamp)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        if self.workload['name'] == 'sysbench':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_sysbench.sh',
                                              self.workload['type'],
                                              self.host,
                                              self.port,
                                              self.user,
                                              150,
                                              800000,
                                              BENCHMARK_WARMING_TIME,
                                              self.threads,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)

        elif self.workload['name'] == 'tpcc':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_tpcc.sh',
                                              self.host,
                                              self.port,
                                              self.user,
                                              self.threads,
                                              BENCHMARK_WARMING_TIME,
                                              BENCHMARK_RUNNING_TIME,
                                              filename,
                                              self.dbname)


        elif self.workload['name'] == 'oltpbench':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_oltpbench.sh',
                                              self.dbname,
                                              self.oltpbench_config_xml,
                                              filename)

        elif self.workload['name'] == 'workload_zoo':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_workload_zoo.sh',
                                              self.workload_zoo_app,
                                              self.workload_zoo_config,
                                              filename)

        elif self.workload['name'] == 'job':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_job.sh',
                                              dirname + '/cli/selectedList.txt',
                                              dirname + '/job_query/queries-mysql-new',
                                              filename,
                                              self.sock
                                              )


        logger.info('[DBG]. {}'.format(cmd))
        return cmd, filename

    def get_states(self, collect_cpu=0):
        start = time.time()
        self.connect_sucess = True
        p = psutil.Process(self.pid)
        if len(p.cpu_affinity())!= CPU_CORE:
            command = 'sudo cgclassify -g memory,cpuset:sever ' + str(self.pid)
            os.system(command)

        internal_metrics = Manager().list()
        im = mp.Process(target=self.get_internal_metrics, args=(internal_metrics,))
        im_alive.value = True
        im.start()
        if collect_cpu:
            rm = ResourceMonitor(self.pid, 1, BENCHMARK_WARMING_TIME, BENCHMARK_RUNNING_TIME)
            rm.run()
        cmd, filename = self.get_benchmark_cmd()
        v = p.cpu_percent()
        print("[{}] benchmark start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=TIMEOUT)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                print("[{}] benchmark finished!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        except subprocess.TimeoutExpired:
            print("[{}] benchmark timeout!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        clear_cmd = """mysqladmin processlist -uroot -S$MYSQL_SOCK | awk '$2 ~ /^[0-9]/ {print "KILL "$2";"}' | mysql -uroot -S$MYSQL_SOCK """
        subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        print("[{}] clear processlist".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        im_alive.value = False
        im.join()
        if collect_cpu:
            rm.terminate()

        if not self.connect_sucess:
            logger.info("connection failed")
            return None
        #try:
        external_metrics = self.get_external_metrics(filename)
        internal_metrics, dirty_pages, hit_ratio, page_data = self._post_handle(internal_metrics)
        logger.info('internal metrics: {}.'.format(list(internal_metrics)))
        ##except:
        #    logger.info("connection failed")
        #    return None
        if collect_cpu:
            monitor_data_dict = rm.get_monitor_data()
            interval = time.time() - start
            avg_read_io = sum(monitor_data_dict['io_read']) / (len(monitor_data_dict['io_read']) + 1e-9)
            avg_write_io = sum(monitor_data_dict['io_write']) / (len(monitor_data_dict['io_write']) + 1e-9)
            avg_virtual_memory = sum(monitor_data_dict['mem_virtual']) / (len(monitor_data_dict['mem_virtual']) + 1e-9)
            avg_physical_memory = sum(monitor_data_dict['mem_physical']) / (
                        len(monitor_data_dict['mem_physical']) + 1e-9)
            resource = (None, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory, dirty_pages, hit_ratio, page_data)
            logger.info(external_metrics)
            return external_metrics, internal_metrics, resource
        else:
            return external_metrics, internal_metrics, [0]*8


    def initialize(self, collect_CPU=0):
        #return np.random.rand(65), np.random.rand(6), np.random.rand(8)
        self.score = 0.
        self.steps = 0
        self.terminate = False
        collect_CPU = True
        logger.info('[DBG]. default tuning knobs: {}'.format(self.default_knobs))
        if self.rds_mode:
            flag = self.apply_rds_knobs(self.default_knobs)
        else:
            flag = self.apply_knobs(self.default_knobs)


        while not flag:
            logger.info('retrying: sleep for {} seconds'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try apply default knobs again')
            if self.rds_mode:
                flag = self.apply_rds_knobs(self.default_knobs)
            else:
                flag = self.apply_knobs(self.default_knobs)

        logger.info('[DBG]. apply default knobs done')

        s = self.get_states(collect_CPU)

        while s == None:
            logger.info('retrying: sleep for {} seconds'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try getting_states again')
            s = self.get_states(collect_CPU)

        external_metrics, internal_metrics, resource = s

        logger.info('[DBG]. get_state done: {}|{}|{}'.format(external_metrics, internal_metrics, resource))

        # TODO(HONG): check external_metrics[0]

        self.last_external_metrics = external_metrics
        self.default_external_metrics = external_metrics
        state = internal_metrics
        save_knobs(self.default_knobs, external_metrics)
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.default_knobs, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]),
                                external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        logger.info("[step {}] default:{}".format(self.step_count, res))
        return state, external_metrics, resource

    def step(self, knobs, episode, step, best_action_applied=False, file=None):
        metrics, internal_metrics, resource = self.step_GP(knobs, best_action_applied)
        try:
            format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
            res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        except:
            format_str = '{}|tps_0|lat_300|qps_0|[]|65d\n'
            res = format_str.format(knobs)
        with open(file, 'a') as f:
            f.write(res)
        reward = self.get_reward(metrics, self.y_variable)
        if not (self.y_variable == 'tps' and metrics[0] <= 0) and not (self.y_variable == 'lat' and metrics[1] <= 0):
            self.last_external_metrics = metrics

        #flag = self._record_best(metrics)
        #if flag:
        #    logger.info('Better performance changed!')
        #else:
        #    logger.info('Performance remained!')
        # get the best performance so far to calculate the reward
        # best_now_performance = self._get_best_now()
        # self.last_external_metrics = best_now_performance

        next_state = internal_metrics
        # TODO(Hong)
        terminate = False

        return reward, next_state, terminate, self.score, metrics

    def get_reward_cpu(self, tps, cpu):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat, tps):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            if _reward < 0 and tps - self.constraint > 0:
                _reward = 0
            if _reward > 0 and tps - self.constraint < 0:
                _reward = 0
            return _reward

        if tps == 0 or cpu == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # latency
        delta_0_lat = float((-cpu + self.default_cpu)) / self.default_cpu
        delta_t_lat = float((-cpu + self.last_cpu)) / self.last_cpu
        reward = calculate_reward(delta_0_lat, delta_t_lat, tps)
        return reward

    def get_reward_cpu_latency(self, tps, cpu, latency):
        """Get the reward that is used in reinforcement learning algorithm.

        The reward is calculated by tps and rt that are external metrics.
        """

        def calculate_reward(delta0, deltat, tps, latency):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            if _reward < 0 and (tps - self.tps_constraint > 0 and latency - self.latency_constraint < 0 ):
                _reward = 0
            if _reward > 0 and (tps - self.tps_constraint < 0 or latency - self.latency_constraint > 0):
                _reward = 0
            return _reward

        if tps == 0 or cpu == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # latency
        delta_0_lat = float((-cpu + self.default_cpu)) / self.default_cpu
        delta_t_lat = float((-cpu + self.last_cpu)) / self.last_cpu
        reward = calculate_reward(delta_0_lat, delta_t_lat, tps, latency)
        return reward


    def step_GP(self, knobs, best_action_applied=False):
        #return np.random.rand(6), np.random.rand(65), np.random.rand(8)

        if self.reinit_interval > 0 and self.reinit_interval % RESTART_FREQUENCY == 0:
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
        self.step_count = self.step_count + 1
        self.reinit_interval = self.reinit_interval + 1
        for key in knobs.keys():
            value = knobs[key]
            if not key in self.knobs_detail.keys() or  not self.knobs_detail[key]['type'] == 'integer':
                continue
            if value > self.knobs_detail[key]['max']:
                knobs[key] = self.knobs_detail[key]['max']
                logger.info("{} with value of is larger than max, adjusted".format(key))
            elif value < self.knobs_detail[key]['min']:
                knobs[key] = self.knobs_detail[key]['min']
                logger.info("{} with value of is smaller than min, adjusted".format(key))
        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))
        collect_resource = False
        if self.rds_mode:
            try:
                flag = self.apply_rds_knobs(knobs)
            except:
                flag = self.apply_knobs(knobs)
        else:
            flag = self.apply_knobs(knobs)

        if not flag:
            if best_action_applied:
                logger.info("[step {}] best:{}|tps_0|lat_300|[]|65d\n".format(self.step_count,knobs))
            else:
                logger.info("[step {}] result:{}|tps_0|lat_300|[]|65d\n".format(self.step_count, knobs))
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
            return [-1, 300, -1 ], np.array([0]*65), 0

        s = self.get_states(collect_resource)

        if s == None:
            if best_action_applied:
                logger.info("[step {}] best:{}|tps_0|[]|65d\n".format(self.step_count,knobs))
            else:
                logger.info("[step {}] result:{}|tps_0|[]|65d\n".format(self.step_count, knobs))
            if self.reinit:
                logger.info('reinitializing database begin')
                self.reinitdb_magic()
                logger.info('database reinitialized')
            return [-1, 300, -1 ], np.array([0]*65), 0

        external_metrics, internal_metrics, resource = s

        '''while external_metrics[0] == 0 or sum(internal_metrics) == 0:
            logger.info('retrying because got invalid metrics. Sleep for {} seconds.'.format(RETRY_WAIT_TIME))
            time.sleep(RETRY_WAIT_TIME)
            logger.info('try get_states again')
            external_metrics, internal_metrics ,resource= self.get_states(collect_resource)
            logger.info('metrics got again, {}|{}'.format(external_metrics, internal_metrics))'''

        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(external_metrics[0]), str(external_metrics[1]), str(external_metrics[2]), external_metrics[3], external_metrics[4],
                                external_metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        if best_action_applied:
            logger.info("[step {}] best:{}".format(self.step_count, res))
        else:
            logger.info("[step {}] result:{}".format(self.step_count, res))
        return external_metrics, internal_metrics, resource


    def step_CobBo(self, **action_cobbo):#**knob_cobbo):
        '''knobs = knob_cobbo.copy()
        for key in knobs.keys():
            knobs[key] = math.floor(knobs[key])
        external_metrics, internal_metrics, resource = self.step_GP(knobs)'''
        action = np.zeros((len(list(action_cobbo.keys()))))
        count = 0
        for key in action_cobbo:
            action[count] = action_cobbo[key]
            count = count + 1
        knobs = generate_knobs(action, 'gp')
        external_metrics, internal_metrics, resource = self.step_GP(knobs)
        return external_metrics[0]

    def terminate(self):
        return False

    def _kill_mysqld(self):
        mysqladmin = os.path.dirname(self.mysqld) + '/mysqladmin'
        cmd = '{} -u{} -S {} shutdown'.format(mysqladmin, self.user, self.sock)
        p_close = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
            ret_code = p_close.poll()
            if ret_code == 0:
                print("Close database successfully")
        except subprocess.TimeoutExpired:
            print("Force close!")
            os.system("ps aux|grep '" + self.sock + "'|awk '{print $2}'|xargs kill -9")
            os.system("ps aux|grep '" + self.mycnf + "'|awk '{print $2}'|xargs kill -9")
        logger.info('mysql is shut down')

    def _start_mysqld(self):
        proc = subprocess.Popen([self.mysqld, '--defaults-file={}'.format(self.mycnf)])
        self.pid = proc.pid
        command = 'sudo cgclassify -g memory,cpuset:sever ' + str(self.pid)
        p = os.system(command)
        if not p:
            logger.info('add {} to memory,cpuset:sever'.format(self.pid))
        else:
            logger.info('Failed: add {} to memory,cpuset:sever'.format(self.pid))
        # os.popen("sudo -S %s" % (command), 'w').write('mypass')
        count = 0
        start_sucess = True
        logger.info('wait for connection')
        error, db_conn = None, None
        while True:
            try:

                dbc = MysqlConnector(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         passwd=self.passwd,
                                         name=self.dbname,
                                         socket=self.sock)
                db_conn = dbc.conn
                if db_conn.is_connected():
                    logger.info('Connected to MySQL database')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 600:
                start_sucess = False
                logger.info("can not connect to DB")
                break

        logger.info('finish {} seconds waiting for connection'.format(count))
        logger.info('{} --defaults-file={}'.format(self.mysqld, self.mycnf))
        logger.info('mysql is up')
        return start_sucess


    def reinitdb_magic(self):
        self._kill_mysqld()
        time.sleep(10)
        os.system('rm -rf {}'.format(self.sock))
        os.system('rm -rf {}'.format(dst_data_path))  # avoid moving src into dst
        logger.info('remove all files in {}'.format(dst_data_path))
        os.system('cp -r {} {}'.format(src_data_path, dst_data_path))
        logger.info('cp -r {} {}'.format(src_data_path, dst_data_path))
        self.pre_combine_log_file_size = log_num_default * log_size_default
        self.apply_knobs(self.default_knobs)
        self.reinit_interval = 0


    def step_SMAC(self, knobs, seed):
        f = open(self.lhs_log, 'a')
        knobs_display = {}
        for key in knobs.keys():
            knobs_display[key] = knobs[key]
        for k in knobs_display.keys():
            if self.knobs_detail[k]['type'] == 'integer' and self.knobs_detail[k]['max'] > sys.maxsize:
                knobs_display[k] = knobs_display[k] * 1000

        logger.info('[SMAC][Episode: 1][Step: {}] knobs generated: {}'.format(self.step_count, knobs_display))
        external_metrics, internal_metrics, resource = self.step_GP(knobs_display)
        record = "{}|{}\n".format(knobs_display, list(internal_metrics))
        f.write(record)
        f.close()
        if self.y_variable == 'tps':
            return -external_metrics[0]
        elif self.y_variable == 'lat':
            return external_metrics[1]

    def step_TPE(self, knobs):
        knobs_display = {}
        for key in knobs.keys():
            if self.knobs_detail[key]['type'] == 'integer':
                knobs_display[key] = int(knobs[key])
                if self.knobs_detail[key]['type'] == 'integer' and self.knobs_detail[key]['max'] > sys.maxsize:
                    knobs_display[key] = knobs_display[key] * 1000
            else:
                knobs_display[key] = knobs[key]

        logger.info('[TPE][Episode: 1][Step: {}] knobs generated: {}'.format(self.step_count, knobs_display))
        external_metrics, internal_metrics, resource = self.step_GP(knobs_display)

        if self.y_variable == 'tps':
            return -external_metrics[0]
        elif self.y_variable == 'lat':
            return external_metrics[1]


    def step_turbo(self, action):#**knob_cobbo):
        knobs = generate_knobs(action, 'gp')
        external_metrics, internal_metrics, resource = self.step_GP(knobs)
        if self.y_variable == 'tps':
            return  -float(external_metrics[0])
        elif self.y_variable == 'lat':
            return float(external_metrics[1])


    def step_openbox(self, config):
        f = open(self.lhs_log, 'a')
        knobs = config.get_dictionary().copy()
        for k in knobs.keys():
            if self.knobs_detail[k]['type'] == 'integer' and  self.knobs_detail[k]['max'] > sys.maxsize:
                knobs[k] = knobs[k] * 1000

        metrics, internal_metrics, resource = self.step_GP(knobs)
        # record = "{}|{}\n".format(knobs, list(internal_metrics))
        # f.write(record)
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()
        if self.y_variable == 'tps':
            return  -float(metrics[0])
        elif self.y_variable == 'lat':
            return float(metrics[1])
