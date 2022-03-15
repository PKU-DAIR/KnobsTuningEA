import os
import time
import pickle
import subprocess
import numpy as np
from .knobs import gen_continuous
from .knobs import logger
from .knobs import ts, knobDF2action, get_data_for_mapping, initialize_knobs, knob2action,  knobDF2action_onehot, gen_continuous_one_hot
from .gp_torch import initialize_GP_model, anlytic_optimize_acqf_and_get_observation, get_acqf
from botorch import fit_gpytorch_model
import torch
import sys
import pdb
import json
from .utils.autotune_exceptions import AutotuneError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from smac.facade.smac_hpo_facade import SMAC4HPO
from .lhs import LHSGenerator
from .utils.parser import convert_65IM_to_51IM, get_action_data_json, get_hist_json
from dynaconf import settings
from .SMAC import SMAC
from smac.initial_design.default_configuration_design import DefaultConfiguration
from hyperopt import fmin, tpe, hp
from autotune.turbo.turbo_m import TurboM
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter
from openbox.optimizer.generic_smbo import SMBO
from autotune.workload_map import WorkloadMapping

RESTART_FREQUENCY = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def save_state_actions(state_action, filename):
    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()




def generate_knobs(action, method):
    if method in ['ddpg', 'ppo', 'sac', 'gp']:
        return gen_continuous(action)
    else:
        raise NotImplementedError("Not implemented generate_knobs")


def load_knobs(knobs_config):
    f = open(knobs_config)
    knob_details = json.load(f)
    knobs = list(knob_details.keys())
    f.close()
    return knob_details





class MySQLTuner:
    def __init__(self, model, env, batch_size, episodes,
                 replay_memory='', idx_sql='', source_data_path='', dst_data_path='', method='DDPG', rl_log='',
                 lhs_log='', restore_state='', workload_map=False,
                 lhs_num=10, y_variable='tps', tr_init=True, trials_file='', rgpe=False,data_repo=''):

        self.model = model
        self.env = env
        self.batch_size = batch_size
        self.episodes = episodes
        if replay_memory:
            self.model.replay_memory.load_memory(replay_memory)
            logger.info('Load memory: {}'.format(self.model.replay_memory))
        self.idx_sql = idx_sql
        self.src = source_data_path
        self.dst = dst_data_path
        self.fine_state_actions = []
        self.train_step = 0
        self.accumulate_loss = [0, 0]
        self.step_counter = 0
        self.expr_name = 'train_{}'.format(ts)
        # ouprocess
        self.sigma = 0.2
        # decay rate
        self.sigma_decay_rate = 0.99
        MySQLTuner.create_output_folders()
        # time for every step, time for training
        self.step_times, self.train_step_times = [], []
        # time for setup + restart + test, time for restart, choose action time
        self.env_step_times, self.env_restart_times, self.action_step_times = [], [], []
        self.noisy = False
        self.method = method
        self.rl_log = rl_log
        self.lhs_log = lhs_log
        self.restore_state = restore_state
        self.workload_map = workload_map
        self.data_repo = data_repo
        self.lhs_num = lhs_num
        self.y_variable = y_variable
        self.trials_file = trials_file
        self.tr_init = tr_init
        if self.workload_map:
            self.mapper = WorkloadMapping(self.data_repo, self.env.knobs_detail, self.y_variable)
        self.rgpe = rgpe



    def tune(self):
        if self.rgpe:
            return self.tune_rgpe()
        elif self.method == 'VBO':
            self.tune_lhs()
            return self.tune_GP_Botorch()
        elif self.method == 'OHBO':
            self.tune_lhs()
            return self.tune_GP_Botorch_onehot()
        elif self.method == 'MBO':
            return self.tune_openbox()
        elif self.method == 'LHS':
            return self.tune_lhs()
        elif self.method == 'DDPG':
            return self.tune_DDPG()
        elif self.method == 'SMAC':
            return self.tune_SMAC()
        elif self.method == 'TPE':
            return self.tune_TPE()
        elif self.method == 'TURBO':
            return self.tune_turbo()
        elif self.method == 'GA':
            return self.tune_gen()
        elif self.method == 'increase':
            return self.incremental_GP()
        elif self.method == 'decrease':
            return self.incremental_Tuneful()



    @staticmethod
    def create_output_folders():
        output_folders = ['log', 'save_memory', 'save_knobs', 'save_state_actions', 'model_params']
        for folder in output_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

    def gen_lhs_samples(self):
        knob_details = self.env.knobs_detail
        lhs_generator = LHSGenerator(self.lhs_num, knob_details)
        lhs_configs = lhs_generator.generate_results()
        return lhs_configs




    def tune_GP_Botorch(self):
        fail_count = 0
        best_action_applied = False
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        '''internal_metrics, metrics, resource = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            metrics[0], metrics[1], metrics[2]
        ))
        # record initial data in res
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.env.default_knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()'''
        best_knob = self.env.default_knobs
        f = open(fn, 'a')
        action_df, df_r, internalm_matrix = get_action_data_json(self.lhs_log)
        if self.y_variable == 'tps':
            Y_variableL = list(df_r[self.y_variable])
        else:
            Y_variableL = list(- df_r[self.y_variable])
        for i in range(0, len(Y_variableL)):
            if Y_variableL[i] <= 0 and self.y_variable == 'tps':
                Y_variableL[i] = min([i for i in Y_variableL if i > 0]) * 0.1
            if Y_variableL[i] >= 0 and self.y_variable == 'lat':
                Y_variableL[i] = min([i for i in Y_variableL if i < 0]) * 10

        action_df = knobDF2action(action_df)  # normalize
        record_num = len(Y_variableL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.double)
        # db_size = self.env.get_db_size()
        # logger.info('Original database size is {}.'.format(db_size))
        NUM_STEP = 200
        # set acquisition func
        acqf_name = 'EI'
        cadidate_size = 1
        reusable = False
        normalized = StandardScaler()
        best_all = max(Y_variableL)
        step_begin = len(Y_variableL)
        self.env.step_count = len(Y_variableL) - 1
        #print("iteration {}: find knobs with {} {}\n".format(step_begin, best_all, self.y_variable))
        for global_step in range(step_begin, NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))
            Y = Y_variableL.copy()
            X = X_scaled.copy()
            if self.workload_map:
                # matched_X_scaled, matched_tps: matched workload
                matched_action_df, matched_y = self.mapper.get_matched_data(X_scaled, internalm_matrix)
                Y = list(matched_y) + Y_variableL
                X = np.vstack((matched_action_df, X_scaled))

            y_scaled = normalized.fit_transform(np.array(Y).reshape(-1, 1))
            train_x = torch.tensor(X.astype('float64')).to(device)
            train_obj = torch.tensor(y_scaled.astype(np.double)).to(device)

            if reusable and model:
                mll, model = initialize_GP_model(
                    train_x,
                    train_obj,
                    model.state_dict()
                )
            else:
                mll, model = initialize_GP_model(train_x, train_obj)
                fit_gpytorch_model(mll)

            acqf = get_acqf(acqf_name, model, train_obj)
            bounds = torch.tensor([[0.0] * X_scaled.shape[1], [1.0] * X_scaled.shape[1]], device=device,
                                  dtype=torch.double)
            best_action_applied = False
            if fail_count > 2:
                self.env.reinitdb_magic()
            if fail_count > 1:
                current_knob = best_knob
                logger.info('best action applied')
                best_action_applied = True
            else:
                new_x = anlytic_optimize_acqf_and_get_observation(acqf, cadidate_size, bounds)
                action = new_x.squeeze(0).cpu().numpy()
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                current_knob = generate_knobs(action, 'gp')
            metrics, internal_metrics, resource = self.env.step_GP(current_knob, best_action_applied)
            #print ((current_knob, metrics))

            X_scaled = np.vstack((X_scaled, new_x.cpu().numpy()))
            if self.workload_map:
                internalm_matrix = np.vstack(
                    (internalm_matrix, internal_metrics.reshape(1, internalm_matrix.shape[1])))

            y_metrics = None
            if metrics[0] < 0:
                logger.info('GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                    global_step, -1, 0, 0, 0))
                format_str = '{}|tps_0|lat_0|qps_0|[]|65d\n'
                res = format_str.format(current_knob)
                if self.y_variable == 'tps':
                    y_metrics = min([i for i in Y_variableL if i > 0]) * 0.9
                else:
                    y_metrics = min([i for i in Y_variableL if i < 0]) * 1.1
                Y_variableL.append(y_metrics)
                f.write(res)
                fail_count = fail_count + 1
                continue
            else:
                fail_count = 0
                if self.y_variable == 'tps':
                    y_metrics = metrics[0]
                else:
                    y_metrics = - metrics[1]
                Y_variableL.append(y_metrics)

            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2], resource[0]))
            format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
            res = format_str.format(current_knob, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                    metrics[3], metrics[4],
                                    metrics[5],
                                    resource[0], resource[1], resource[2], resource[3], resource[4],
                                    resource[5], resource[6], resource[7], list(internal_metrics))
            f.write(res)
            if y_metrics > best_all:
                best_all = y_metrics
                best_knob = current_knob
            # TODO with enough data, no workload mapping
            if len(Y_variableL) >= 50:
                self.workload_map = False

        f.close()
        return




    def tune_lhs(self):
        if self.lhs_num == 0:
            return
        lhs_configs = self.gen_lhs_samples()
        for step, knobs in enumerate(lhs_configs):
            if self.lhs_log != '':
                fn = self.lhs_log
            else:
                fn = 'gp_data.res'
            f = open(fn, 'a')
            metrics, internal_metrics, resource = self.env.step_GP(knobs)
            #print ((knobs, metrics))
            if metrics[0] < 0:
                logger.info('[LHS][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                    step+1, -1, 0, 0, 0))
                format_str = '{}|tps_{}|[]|65d\n'
                res = format_str.format(knobs,  -np.random.random())
                f.write(res)
                continue

            logger.info('[LHS][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                step+1, metrics[0], metrics[1], metrics[2], resource[0]))
            format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
            res = format_str.format(knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                    metrics[3], metrics[4],
                                    metrics[5],
                                    resource[0], resource[1], resource[2], resource[3], resource[4],
                                    resource[5], resource[6], resource[7], list(internal_metrics))
            f.write(res)


    def tune_DDPG(self):
        fail_count = False
        global_t = 0
        best_tps = 0

        best_action_applied = False
        for episode in range(self.episodes):
            logger.info('env initializing')
            current_state, metrics, _ = self.env.initialize()
            logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
                metrics[0], metrics[1], metrics[2]
            ))
            self.model.reset(self.sigma)
            t = 0
            while True:
                logger.info('entering episode{}s step{}'.format(episode, t))
                state = current_state
                if self.noisy:
                    self.model.sample_noise()
                action_step_time = time.time()
                best_action_applied = False
                if fail_count > 2:
                    self.env.reinitdb_magic()
                if fail_count > 1:
                    if  not best_action is None:
                        action = best_action
                        logger.info('best action applied')
                        best_action_applied = True
                    else:
                        action = np.random.rand(self.model.n_actions)
                        logger.info('random action applied')

                elif np.random.random() < 0.7:  # avoid too nus reward in the fisrt 100 step
                    action = self.model.choose_action(state, 1 / (global_t + 1))
                        #logger.info('[ddpg] Action: {}'.format(action))
                else:
                    action = self.model.choose_action(state, 1)
                        #logger.info('[ddpg] Action: {}'.format(action))

                logger.info('[ddpg] Action: {}'.format(action))
                current_knob = generate_knobs(action, 'ddpg')
                reward, state_, done, score, metrics = self.env.step(current_knob, episode, t, best_action_applied, self.lhs_log)
                if metrics[0] > best_tps:
                    best_tps = metrics[0]
                    best_action = action
                logger.info('[ddpg][Episode: {}][Step: {}][Metric tps:{} lat:{} qps:{}][Reward: {} Score: {}]'.format(
                    episode, t, metrics[0], metrics[1], metrics[2], reward, score
                ))

                if metrics[0] < 0:
                    next_state = state
                    fail_count = fail_count + 1
                else:
                    next_state = state_
                    fail_count = 0

                self.model.add_sample(state, action, reward, next_state, done)

                if reward > 10:
                    self.fine_state_actions.append((state, action))

                current_state = next_state

                if len(self.model.replay_memory) > self.batch_size:
                    losses = []

                    for _ in range(4):
                        losses.append(self.model.update())
                        self.train_step += 1


                    self.accumulate_loss[0] += sum(x[0] for x in losses)
                    self.accumulate_loss[1] += sum(x[1] for x in losses)

                    logger.info('[ddpg][Episode: {}][Step: {}] Critic: {} Actor: {})'.format(
                        episode, t, self.accumulate_loss[0] / self.train_step, self.accumulate_loss[1] / self.train_step
                    ))

                t += 1
                logger.info('The end of global step {}.'.format(global_t))
                global_t += 1
                self.step_counter += 1

                # save replay memory
                if self.step_counter % 10 == 0:
                    self.model.replay_memory.save('save_memory/{}.pkl'.format(self.expr_name))
                    save_state_actions(self.fine_state_actions, 'save_state_actions/{}.pkl'.format(self.expr_name))

                if self.step_counter % 5 == 0:
                    self.model.save_model('model_params', title='{}_{}'.format(self.expr_name, self.step_counter))

                if t >= 100:
                    done = True
                if done or score < -50:
                    break




    def tune_SMAC(self):
        # Create defaults
        rh = None
        stats = None
        incumbent = None
        tuner_SMAC = SMAC(self.env.knobs_detail)
        tuner_SMAC.init_Configuration()
        ts = int(time.time())
        if self.restore_state is not '':
            rh, stats, incumbent, tuner_SMAC.scenario = tuner_SMAC.restore(self.restore_state, ts, self.load_num)

        smac = SMAC4HPO(
                scenario=tuner_SMAC.scenario,
                initial_design=DefaultConfiguration,
                tae_runner=self.env.step_SMAC,
                runhistory=rh,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=ts)

        self.env.lhs_log = self.lhs_log
        if self.workload_map:
            #smac.__version__ == 0.12.0
            def _collect_data_to_train_model1():
                flag, matched_runhistory = self.mapper.get_runhistory_smac(self.lhs_log, tuner_SMAC.cs)
                emp = smac.solver.epm_chooser

            # if we use a float value as a budget, we want to train the model only on the highest budget
                available_budgets = []
                for i in emp.runhistory.data.keys():
                    available_budgets.append(i.budget)

                # Sort available budgets from highest to lowest budget
                available_budgets = sorted(list(set(available_budgets)), reverse=True)

            # Get #points per budget and if there are enough samples, then build a model
                for b in available_budgets:
                    X, Y = emp.rh2EPM.transform(emp.runhistory, budget_subset=[b, ])
                    if flag:
                        X_matched, Y_matched = emp.rh2EPM.transform(matched_runhistory , budget_subset=[b, ])
                        X = np.vstack((X, X_matched))
                        Y = np.vstack((Y, Y_matched))
                    ##生成mapped的runhistory
                    #转化为X，Y
                    if X.shape[0] >= emp.min_samples_model:
                        emp.currently_considered_budgets = [b, ]
                        return X, Y

                return np.empty(shape=[0, 0]), np.empty(shape=[0, ])

            smac.solver.epm_chooser._collect_data_to_train_model = _collect_data_to_train_model1

            #smac.optimizer.epm_configuration_chooser.EPMChooser

        # Example call of the function with default values
        # It returns: Status, Cost, Runtime, Additional Infos
        #smac.solver.intensifier.tae_runner.use_pynisher = False
        # Start optimization
        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
        logger.info("Optimized Value: %.2f" % inc_value)

    def tune_TPE(self):
        space = {}
        KNOBS = self.env.knobs_detail.keys()
        default_knob = {}
        self.env.lhs_log = self.lhs_log
        for knob in KNOBS:
            value = self.env.knobs_detail[knob]
            knob_type = value['type']
            if knob_type == 'enum':
                #pdb.set_trace()
                space[knob] = hp.choice(knob, value["enum_values"])
                #try:
                default_knob[knob] = value["enum_values"].index(str(self.env.default_knobs[knob]))
                #except:
                #    pdb.set_trace()
            elif knob_type == 'integer':
                if self.env.knobs_detail[knob]['max'] > sys.maxsize:
                    min_val, max_val = int(value['min']/1000), int(value['max']/1000)
                    space[knob] = hp.quniform(knob, min_val, max_val, q=1)
                    default_knob[knob] = int(self.env.default_knobs[knob] /1000)
                    continue
                min_val, max_val = value['min'], value['max']
                if value.get('stride'):
                    space[knob] = hp.quniform(knob, min_val, max_val, q=value['stride'])
                else:
                    space[knob] = hp.quniform(knob, min_val, max_val, q=1)
                default_knob[knob] = self.env.default_knobs[knob]
            elif knob_type == 'float':
                min_val, max_val = value['min'], value['max']
                space[knob] = hp.uniform(knob, min_val, max_val)
                default_knob[knob] = self.env.default_knobs[knob]
        #pdb.set_trace()
        best_knobs = fmin(
            fn=self.env.step_TPE,
            space=space,
            algo=tpe.suggest,
            max_evals=210,
            trials_save_file=self.trials_file,
            points_to_evaluate=[default_knob]
        )
        logger.info("Best knobs so far: {}".format(best_knobs))

    def tune_turbo(self):
        self.env.lhs_log = self.lhs_log
        internal_metrics, metrics, resource = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            metrics[0], metrics[1], metrics[2]
        ))
        dim = len(list(self.env.knobs_detail.keys()))
        #pdb.set_trace()
        lb = 0 * np.ones(dim)
        ub = 1 * np.ones(dim)
        turbo_m = TurboM(
            f=self.env.step_turbo,  # Handle to objective function
            lb=lb,  # Numpy array specifying lower bounds
            ub=ub,  # Numpy array specifying upper bounds
            n_init=self.lhs_num,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=200,  # Maximum number of evaluations
            n_trust_regions=5,  # Number of trust regions
            batch_size=1,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
            init_flag = self.tr_init,
            y_variable = self.y_variable
        )

        turbo_m.optimize()

        X = turbo_m.X  # Evaluated points
        fX = turbo_m.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]

        print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))


    def tune_openbox(self):
        KNOBS = self.env.knobs_detail
        config_space = ConfigurationSpace()
        for name in  KNOBS.keys():
            value = KNOBS[name]
            knob_type = value['type']
            if knob_type == 'enum':
                knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]], default_value=str(value['default']))
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if self.env.knobs_detail[name]['max'] > sys.maxsize:
                    knob = UniformIntegerHyperparameter(name, int(min_val/1000), int(max_val/1000), default_value=int(value['default']/1000))
                else:
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
            elif knob_type == 'float':
                min_val, max_val = value['min'], value['max']
                knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
            config_space.add_hyperparameters([knob])

        task_id = self.task_id
        bo = SMBO(self.env.step_openbox,
           config_space,
           num_objs=1,
           num_constraints=0,
           max_runs=210,
           surrogate_type='gp',
           acq_optimizer_type='local_random',#'random_scipy',#
           initial_runs=10,
           init_strategy='random_explore_first',
           task_id=task_id,
           time_limit_per_trial=60*200)

        save_file = task_id + '.pkl'
        advisor = bo.config_advisor

        def load_history_from_pickle(records_num=-1):
            with open(save_file, 'rb') as f:
                history_container_old = pickle.load(f)
            if records_num == -1:
                advisor.history_container = history_container_old
                print("Load all the history records from {}".format(save_file))
                bo.iteration_id = len(history_container_old.configurations)
                return history_container_old

            for i in range(records_num):
                bo.iteration_id = bo.iteration_id + 1
                config = history_container_old.configurations[i]
                trial_state = history_container_old.trial_states[i]
                constraints = history_container_old.constraint_perfs[i]
                if advisor.num_objs == 1:
                    objs = [history_container_old.perfs[i]]
                else:
                    objs = history_container_old.perfs[i]
                elapsed_time = history_container_old.elapsed_times[i]
                observation = config, trial_state, constraints, objs, elapsed_time

                advisor.update_observation(observation)

            print("Load {} history recorde from {}".format(records_num, save_file))
            return advisor.history_container

        #if os.path.exists(save_file):
        #    load_history_from_pickle(self.load_num)

        self.env.lhs_log = self.lhs_log
        #bo.config_advisor.load_history_from_pickle(-1)
        if self.workload_map:
            epm = bo.config_advisor.surrogate_model
            def train1(X, Y):
                self.types = epm._initial_types.copy()

                if len(X.shape) != 2:
                    raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
                if X.shape[1] != len(epm.types):
                    raise ValueError(
                        'Feature mismatch: X should have %d features, but has %d' % (X.shape[1], len(epm.types)))
                if X.shape[0] != Y.shape[0]:
                    raise ValueError('X.shape[0] (%s) != y.shape[0] (%s)' % (X.shape[0], Y.shape[0]))

                epm.n_params = X.shape[1] - epm.n_feats
                X_matched, Y_matched = self.mapper.get_XY_openbox(self.lhs_log, config_space)
                scaler = StandardScaler()
                Y_matched = scaler.fit_transform(Y_matched.reshape(-1,1))
                Y = scaler.fit_transform(Y.reshape(-1, 1))
                X = np.vstack((X, X_matched))
                Y = np.vstack((Y.reshape(-1, 1), Y_matched.reshape(-1, 1)))
                epm.normalize_y = False
                return epm._train(X, Y)

            bo.config_advisor.surrogate_model.train = train1

        history = bo.run()
        '''try:
            history = bo.run()
        except:
            with open(save_file, 'wb') as f:
                pickle.dump(bo.config_advisor.history_container, f)
                print("Save history recorde to {}".format(save_file))'''


    def tune_rgpe(self):
        task_id = self.task_id

        KNOBS = self.env.knobs_detail
        config_space = ConfigurationSpace()
        for name in KNOBS.keys():
            value = KNOBS[name]
            knob_type = value['type']
            if knob_type == 'enum':
                enum_values = []
                for i in value['enum_values']:
                    try:
                        ev = int(i)
                    except:
                        ev = i
                    enum_values.append(ev)
                try:
                    dv = int(value['default'])
                except:
                    dv = value['default']
                knob = CategoricalHyperparameter(name, enum_values, default_value=dv)
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if name == "innodb_online_alter_log_max_size":
                    knob = UniformIntegerHyperparameter(name, int(min_val / 10), int(max_val / 10),
                                                        default_value=int(value['default'] / 10))
                else:
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])

            config_space.add_hyperparameters([knob])

        odL = []
        files = os.listdir(self.data_repo)
        for f in files:
            try:
                od = get_hist_json(self.data_repo + f, cs=config_space, y_variable=self.y_variable)
                odL.append(od)
            except:
                logger.info('build base surrogate failed for {}'.format(f))

        if self.method == 'MBO':
            rgpe_method = 'rgpe_gp'
        else:
            self.method = 'rgpe_prf'
        surrogate_type = 'tlbo_rgpe_' + rgpe_method

        bo = SMBO(self.env.step_openbox,
                  config_space,
                  num_objs=1,
                  num_constraints=0,
                  max_runs=1000,
                  surrogate_type=surrogate_type,
                  history_bo_data=odL,
                  acq_optimizer_type='local_random',
                  initial_runs=10,
                  init_strategy='random_explore_first',
                  task_id=task_id,
                  time_limit_per_trial=60 * 200)

        history = bo.run()



    def load_data(self, default_knobs, tuneful=False):
        action_df, df_r = get_increment_result(self.lhs_log, default_knobs)
        ind = df_r["tps"].idxmax()
        knobs = action_df.iloc[ind, :]
        knobs_best = {}
        #pdb.set_trace()
        for k in knobs.index:
            knobs_best[k] = knobs[k]
        if self.y_variable == 'tps':
            Y_variableL = list(df_r[self.y_variable])
        else:
            Y_variableL = list(- df_r[self.y_variable])
        for i in range(0, len(Y_variableL)):
            if Y_variableL[i] <= 0 and self.y_variable == 'tps':
                Y_variableL[i] = min([i for i in Y_variableL if i > 0]) * 0.1
            if Y_variableL[i] >= 0 and self.y_variable == 'lat':
                Y_variableL[i] = min([i for i in Y_variableL if i < 0]) * 10

        action_df = knobDF2action(action_df)  # normalize
        record_num = len(Y_variableL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.double)


        best_all = max(Y_variableL)
        step_begin = len(Y_variableL)
        self.env.step_count = len(Y_variableL) - 1
        print("iteration {}: find knobs with {} {}\n".format(step_begin, best_all, self.y_variable))

        return  X_scaled, Y_variableL, knobs_best

    def incremental_GP(self):
        fail_count = 0
        NUM_STEP = 12
        acqf_name = 'EI'
        cadidate_size = 1

        f = open(self.lhs_log, 'a')
        internal_metrics, metrics, resource = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            metrics[0], metrics[1], metrics[2]
        ))
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.env.default_knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()

        normalized = StandardScaler()
        for iter in range(0, 17):
            knobs_tune = {}
            for i in range(4 + iter * 2):
                key = list(self.env.default_knobs.keys())[i]
                knobs_tune[key] = self.env.default_knobs[key]

            #pdb.set_trace()
            initialize_knobs(self.env.knobs_config, 4 + iter * 2)
            X_scaled, Y_variableL, _ = self.load_data(knobs_tune, tuneful=True)
            f = open(self.lhs_log, 'a')
            for global_step in range(0, NUM_STEP):
                logger.info('entering episode 0 step {}'.format(global_step))
                Y = Y_variableL.copy()
                X = X_scaled.copy()
                y_scaled = normalized.fit_transform(np.array(Y).reshape(-1, 1))
                train_x = torch.tensor(X.astype('float64'))
                train_obj = torch.tensor(y_scaled.astype(np.double))
                mll, model = initialize_GP_model(train_x, train_obj)
                fit_gpytorch_model(mll)

                acqf = get_acqf(acqf_name, model, train_obj)
                bounds = torch.tensor([[0.0] * X_scaled.shape[1], [1.0] * X_scaled.shape[1]], device=device,
                                  dtype=torch.double)
                best_action_applied = False

                new_x = anlytic_optimize_acqf_and_get_observation(acqf, cadidate_size, bounds)
                action = new_x.squeeze(0).cpu().numpy()
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                current_knob = generate_knobs(action, 'gp')
                metrics, internal_metrics, resource = self.env.step_GP(current_knob, best_action_applied)

                X_scaled = np.vstack((X_scaled, new_x.numpy()))

                if metrics[0] < 0:
                    logger.info('GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                    global_step, -1, 0, 0, 0))
                    format_str = '{}|tps_0|lat_0|qps_0|[]|65d\n'
                    res = format_str.format(current_knob)
                    if self.y_variable == 'tps':
                        y_metrics = min([i for i in Y_variableL if i > 0]) * 0.9
                    else:
                        y_metrics = min([i for i in Y_variableL if i < 0]) * 1.1
                    Y_variableL.append(y_metrics)
                    f.write(res)
                    fail_count = fail_count + 1
                    continue
                else:
                    fail_count = 0
                    if self.y_variable == 'tps':
                        y_metrics = metrics[0]
                    else:
                        y_metrics = - metrics[1]
                    Y_variableL.append(y_metrics)

                logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2], resource[0]))
                format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
                res = format_str.format(current_knob, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                    metrics[3], metrics[4],
                                    metrics[5],
                                    resource[0], resource[1], resource[2], resource[3], resource[4],
                                    resource[5], resource[6], resource[7], list(internal_metrics))
                f.write(res)

            f.close()
        return


    def incremental_Tuneful(self):
        fail_count = 0
        NUM_STEP = 20
        acqf_name = 'EI'
        cadidate_size = 1

        f = open(self.lhs_log, 'a')
        internal_metrics, metrics, resource = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            metrics[0], metrics[1], metrics[2]
        ))
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.env.default_knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()

        normalized = StandardScaler()
        num_tuned = 196
        for iter in range(0, 10):
            knobs_tune = {}
            for i in range(num_tuned):
                key = list(self.env.default_knobs.keys())[i]
                knobs_tune[key] = self.env.default_knobs[key]

            #pdb.set_trace()
            initialize_knobs(self.env.knobs_config, num_tuned)
            import math
            num_tuned = math.ceil(num_tuned * 0.6)
            X_scaled, Y_variableL, knobs_best = self.load_data(knobs_tune)
            f = open(self.lhs_log, 'a')
            self.env.apply_knobs(knobs_best)
            for global_step in range(0, NUM_STEP):
                logger.info('entering episode 0 step {}'.format(global_step))

                Y = Y_variableL.copy()
                X = X_scaled.copy()
                y_scaled = normalized.fit_transform(np.array(Y).reshape(-1, 1))
                train_x = torch.tensor(X.astype('float64'))
                train_obj = torch.tensor(y_scaled.astype(np.double))
                mll, model = initialize_GP_model(train_x, train_obj)
                fit_gpytorch_model(mll)

                acqf = get_acqf(acqf_name, model, train_obj)
                bounds = torch.tensor([[0.0] * X_scaled.shape[1], [1.0] * X_scaled.shape[1]], device=device,
                                  dtype=torch.double)
                best_action_applied = False

                new_x = anlytic_optimize_acqf_and_get_observation(acqf, cadidate_size, bounds)
                action = new_x.squeeze(0).cpu().numpy()
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                current_knob = generate_knobs(action, 'gp')
                metrics, internal_metrics, resource = self.env.step_GP(current_knob, best_action_applied)

                X_scaled = np.vstack((X_scaled, new_x.numpy()))

                if metrics[0] < 0:
                    logger.info('GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                    global_step, -1, 0, 0, 0))
                    format_str = '{}|tps_0|lat_0|qps_0|[]|65d\n'
                    res = format_str.format(current_knob)
                    if self.y_variable == 'tps':
                        y_metrics = min([i for i in Y_variableL if i > 0]) * 0.9
                    else:
                        y_metrics = min([i for i in Y_variableL if i < 0]) * 1.1
                    Y_variableL.append(y_metrics)
                    f.write(res)
                    fail_count = fail_count + 1
                    continue
                else:
                    fail_count = 0
                    if self.y_variable == 'tps':
                        y_metrics = metrics[0]
                    else:
                        y_metrics = - metrics[1]
                    Y_variableL.append(y_metrics)

                logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2], resource[0]))
                format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
                res = format_str.format(current_knob, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                    metrics[3], metrics[4],
                                    metrics[5],
                                    resource[0], resource[1], resource[2], resource[3], resource[4],
                                    resource[5], resource[6], resource[7], list(internal_metrics))
                f.write(res)

            f.close()
        return



    def tune_gen(self):
        KNOBS = self.env.knobs_detail
        config_space = ConfigurationSpace()
        for name in  KNOBS.keys():
            value = KNOBS[name]
            knob_type = value['type']
            if knob_type == 'enum':
                knob = CategoricalHyperparameter(name, [str(i) for i in value["enum_values"]], default_value=str(value['default']))
            elif knob_type == 'integer':
                min_val, max_val = value['min'], value['max']
                if self.env.knobs_detail[name]['max'] > sys.maxsize:
                    knob = UniformIntegerHyperparameter(name, int(min_val/1000), int(max_val/1000), default_value=int(value['default']/1000))
                else:
                    knob = UniformIntegerHyperparameter(name, min_val, max_val, default_value=value['default'])
            elif knob_type == 'float':
                min_val, max_val = value['min'], value['max']
                knob = UniformFloatHyperparameter(name, min_val, max_val, default_value=value['default'])
            config_space.add_hyperparameters([knob])

        task_id = self.task_id
        bo = SMBO(self.env.step_openbox,
           config_space,
           num_objs=1,
           num_constraints=0,
           advisor_type='ea',
           initial_runs=10,
           task_id=task_id,
           time_limit_per_trial=60*200)

        save_file = task_id + '.pkl'
        advisor = bo.config_advisor

        def load_history_from_pickle(records_num=-1):
            with open(save_file, 'rb') as f:
                history_container_old = pickle.load(f)
            if records_num == -1:
                advisor.history_container = history_container_old
                print("Load all the history records from {}".format(save_file))
                bo.iteration_id = len(history_container_old.configurations)
                return history_container_old

            for i in range(records_num):
                bo.iteration_id = bo.iteration_id + 1
                config = history_container_old.configurations[i]
                trial_state = history_container_old.trial_states[i]
                constraints = history_container_old.constraint_perfs[i]
                if advisor.num_objs == 1:
                    objs = [history_container_old.perfs[i]]
                else:
                    objs = history_container_old.perfs[i]
                elapsed_time = history_container_old.elapsed_times[i]
                observation = config, trial_state, constraints, objs, elapsed_time

                advisor.update_observation(observation)

            print("Load {} history recorde from {}".format(records_num, save_file))
            return advisor.history_container

        #if os.path.exists(save_file):
        #    load_history_from_pickle(self.load_num)

        self.env.lhs_log = self.lhs_log
        #bo.config_advisor.load_history_from_pickle(-1)
        if self.workload_map:
            epm = bo.config_advisor.surrogate_model
            def train1(X, Y):
                self.types = epm._initial_types.copy()

                if len(X.shape) != 2:
                    raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
                if X.shape[1] != len(epm.types):
                    raise ValueError(
                        'Feature mismatch: X should have %d features, but has %d' % (X.shape[1], len(epm.types)))
                if X.shape[0] != Y.shape[0]:
                    raise ValueError('X.shape[0] (%s) != y.shape[0] (%s)' % (X.shape[0], Y.shape[0]))

                epm.n_params = X.shape[1] - epm.n_feats
                X_matched, Y_matched = self.mapper.get_XY_openbox(self.lhs_log, config_space)
                scaler = StandardScaler()
                Y_matched = scaler.fit_transform(Y_matched.reshape(-1,1))
                Y = scaler.fit_transform(Y.reshape(-1, 1))
                X = np.vstack((X, X_matched))
                Y = np.vstack((Y.reshape(-1, 1), Y_matched.reshape(-1, 1)))
                epm.normalize_y = False
                return epm._train(X, Y)

            bo.config_advisor.surrogate_model.train = train1

        history = bo.run()
        try:
            history = bo.run()
        except:
            with open(save_file, 'wb') as f:
                pickle.dump(bo.config_advisor.history_container, f)
                print("Save history recorde to {}".format(save_file))


    def tune_GP_Botorch_onehot(self):
        feature_len = 0
        for k in self.env.knobs_detail.keys():
            if self.env.knobs_detail[k]['type'] == 'enum':
                feature_len = feature_len + len(self.env.knobs_detail[k]['enum_values'])
            else:
                feature_len = feature_len + 1

        fail_count = 0
        best_action_applied = False
        if self.lhs_log != '':
            fn = self.lhs_log
        else:
            fn = 'gp_data.res'
        f = open(fn, 'a')
        internal_metrics, metrics, resource = self.env.initialize()
        logger.info('[Env initialized][Metrics tps:{} lat: {} qps: {}]'.format(
            metrics[0], metrics[1], metrics[2]
        ))
        # record initial data in res
        format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
        res = format_str.format(self.env.default_knobs, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                metrics[3], metrics[4],
                                metrics[5],
                                resource[0], resource[1], resource[2], resource[3], resource[4],
                                resource[5], resource[6], resource[7], list(internal_metrics))
        f.write(res)
        f.close()
        best_knob = self.env.default_knobs
        f = open(fn, 'a')
        action_df, df_r, internalm_matrix = get_action_data_json(self.lhs_log)
        if self.y_variable == 'tps':
            Y_variableL = list(df_r[self.y_variable])
        else:
            Y_variableL = list(- df_r[self.y_variable])
        for i in range(0, len(Y_variableL)):
            if Y_variableL[i] <= 0 and self.y_variable == 'tps':
                Y_variableL[i] = min([i for i in Y_variableL if i > 0]) * 0.1
            if Y_variableL[i] >= 0 and self.y_variable == 'lat':
                Y_variableL[i] = min([i for i in Y_variableL if i < 0]) * 10

        action_df = knobDF2action_onehot(action_df)  # normalize
        record_num = len(Y_variableL)
        X_scaled = action_df[:record_num, :]
        X_scaled = X_scaled.astype(np.double)
        # db_size = self.env.get_db_size()
        # logger.info('Original database size is {}.'.format(db_size))
        NUM_STEP = 200
        # set acquisition func
        acqf_name = 'EI'
        cadidate_size = 1
        reusable = False
        normalized = StandardScaler()
        best_all = max(Y_variableL)
        step_begin = len(Y_variableL)
        self.env.step_count = len(Y_variableL) - 1
        #print("iteration {}: find knobs with {} {}\n".format(step_begin, best_all, self.y_variable))
        for global_step in range(step_begin, NUM_STEP):
            logger.info('entering episode 0 step {}'.format(global_step))
            Y = Y_variableL.copy()
            X = X_scaled.copy()
            if self.workload_map:
                # matched_X_scaled, matched_tps: matched workload
                matched_action_df, matched_y = self.mapper.get_matched_data(X_scaled, internalm_matrix)
                Y = list(matched_y) + Y_variableL
                X = np.vstack((matched_action_df, X_scaled))

            y_scaled = normalized.fit_transform(np.array(Y).reshape(-1, 1))
            train_x = torch.tensor(X.astype('float64')).to(device)
            train_obj = torch.tensor(y_scaled.astype(np.double)).to(device)

            if reusable and model:
                mll, model = initialize_GP_model(
                    train_x,
                    train_obj,
                    model.state_dict()
                )
            else:
                mll, model = initialize_GP_model(train_x, train_obj)
                mll, model = initialize_GP_model(train_x, train_obj)
                fit_gpytorch_model(mll)

            acqf = get_acqf(acqf_name, model, train_obj)
            bounds = torch.tensor([[0.0] * feature_len, [1.0] * feature_len], device=device,
                                  dtype=torch.double)
            best_action_applied = False
            if fail_count > 2:
                self.env.db.reinitdb_magic()
            if fail_count > 1:
                current_knob = best_knob
                logger.info('best action applied')
                best_action_applied = True
            else:
                new_x = anlytic_optimize_acqf_and_get_observation(acqf, cadidate_size, bounds)
                action = new_x.squeeze(0).cpu().numpy()
                logger.info('[GP-BOTORCH] Action: {}'.format(action))
                current_knob = gen_continuous_one_hot(action)
            metrics, internal_metrics, resource = self.env.step_GP(current_knob, best_action_applied)

            X_scaled = np.vstack((X_scaled, new_x.cpu().numpy()))
            if self.workload_map:
                internalm_matrix = np.vstack(
                    (internalm_matrix, internal_metrics.reshape(1, internalm_matrix.shape[1])))

            y_metrics = None
            if metrics[0] < 0:
                logger.info('GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                    global_step, -1, 0, 0, 0))
                format_str = '{}|tps_0|lat_0|qps_0|[]|65d\n'
                res = format_str.format(current_knob)
                if self.y_variable == 'tps':
                    y_metrics = min([i for i in Y_variableL if i > 0]) * 0.9
                else:
                    y_metrics = min([i for i in Y_variableL if i < 0]) * 1.1
                Y_variableL.append(y_metrics)
                f.write(res)
                fail_count = fail_count + 1
                continue
            else:
                fail_count = 0
                if self.y_variable == 'tps':
                    y_metrics = metrics[0]
                else:
                    y_metrics = - metrics[1]
                Y_variableL.append(y_metrics)

            logger.info('[GP-BOTORCH][Episode: 1][Step: {}][Metric tps:{} lat:{} qps:{} cpu:{}]'.format(
                global_step, metrics[0], metrics[1], metrics[2], resource[0]))
            format_str = '{}|tps_{}|lat_{}|qps_{}|tpsVar_{}|latVar_{}|qpsVar_{}|cpu_{}|readIO_{}|writeIO_{}|virtaulMem_{}|physical_{}|dirty_{}|hit_{}|data_{}|{}|65d\n'
            res = format_str.format(current_knob, str(metrics[0]), str(metrics[1]), str(metrics[2]),
                                    metrics[3], metrics[4],
                                    metrics[5],
                                    resource[0], resource[1], resource[2], resource[3], resource[4],
                                    resource[5], resource[6], resource[7], list(internal_metrics))
            f.write(res)
            if y_metrics > best_all:
                best_all = y_metrics
                best_knob = current_knob
            # TODO with enough data, no workload mapping
            if len(Y_variableL) >= 50:
                self.workload_map = False

        f.close()
        return
