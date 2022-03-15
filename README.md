# A Comprehensive Experimental Evaluation for Database Configuration Tuning

This is the source code to the paper ["Facilitating Database Tuning with Hyper-ParameterOptimization: A Comprehensive Experimental Evaluation"](https://arxiv.org/abs/2110.12654). Please refer to the paper for the experimental details.

## Table of Content

* [An Efficient Database Configuration Tuning Benchmark via Surrogate](https://github.com/PKU-DAIR/KnobsTuningEA/blob/main/README.md#an-efficient-database-configuration-tuning-benchmark-via-surrogate)
* [Experimental Evaluation](https://github.com/PKU-DAIR/KnobsTuningEA#experimental-evaluation)
  * [Environment Installation](https://github.com/PKU-DAIR/KnobsTuningEA#environment-installation)
  * [Workload Preparation](https://github.com/PKU-DAIR/KnobsTuningEA#workload-preparation)
  * [Environment Variables](https://github.com/PKU-DAIR/KnobsTuningEA#environment-variables)
  * [Experiments Design](https://github.com/PKU-DAIR/KnobsTuningEA#experiments-design)
    * [Exp.1: Tuning improvement over knob set generated by different important measurements.](https://github.com/PKU-DAIR/KnobsTuningEA#exp1-tuning-improvement-over-knob-set-generated-by-different-important-measurements)
    * [Exp.2: Performance improvement and tuning cost when increasing the number of tuned knobs.](https://github.com/PKU-DAIR/KnobsTuningEA#exp2-performance-improvement-and-tuning-cost-when-increasing-the-number-of-tuned-knobs)
    * [Exp.3: Incremental Knob Selection.](https://github.com/PKU-DAIR/KnobsTuningEA#exp3-incremental-knob-selection)
    * [Exp.4: Optimizer comparision on different configuration space.](https://github.com/PKU-DAIR/KnobsTuningEA#exp4-optimizer-comparision-on-different-configuration-space)
    * [Exp.5: Comparison experiment for knobs heterogeneity.](https://github.com/PKU-DAIR/KnobsTuningEA#exp5-comparison-experiment-for-knobs-heterogeneity)
    * [Exp.6: Algorithm overhead comparison.](https://github.com/PKU-DAIR/KnobsTuningEA#exp6-algorithm-overhead-comparison)
    * [Exp.7: Transfering methods comparison.](https://github.com/PKU-DAIR/KnobsTuningEA#exp7-transfering-methods-comparison)

- [Project Code Overview](https://github.com/PKU-DAIR/KnobsTuningEA#project-code-overview)

  

## An Efficient Database Configuration Tuning Benchmark via Surrogate
Through the benchmark, you can evaluate the tuning optimizers' performance with minimum overhead.
 

### Quick installation & Run

1. Preparations: Python == 3.7

2. Install packages and download the surrogate model

   ```shell
   pip install -r requirements.txt
   pip install .
   ```
  The surrogate models can be found in [the Google drive](https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing). To easily run the tuning benchmark, you can download the surrogate models and place them in the fold autotune/tuning_benchmark/surrogate.

3. Run the benchmark for knob selection. We use selecting 5 knobs and  tuning JOB via SMAC as an example.
```shell
bash experiment/compare_knob_selection_pg.sh JOB SMAC 5
   ```
4. Run the  benchmark for optimizer. We use optimization over the configuration space of JOB as an example. You need to set the model_path.
	
```shell
python run_benchmark.py --method=VBO --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job  --lhs_log=result/job_5knobs_vbo.res --model
python run_benchmark.py --method=MBO   --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job --lhs_log=result/job_5knobs_mbo.res
python run_benchmark.py --method=SMAC  --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job   --lhs_log=result/job_5knobs_smac.res
python run_benchmark.py --method=TPE --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job  --lhs_log=result/job_5knobs_tpe.res
python run_benchmark.py --method=TURBO --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job --lhs_log=result/job_5knobs_turbo.res --tr_init 
python run_benchmark.py --method=GA --knobs_config=experiment/gen_knobs/JOB_shap.json --knobs_num=5 --workload=job --lhs_log=result/job_5knobs_ga.res 
   ```
### Data Description
You can find all the training data for the tuning benchmark in autotune/tuning_benchmark/data. 



## Experimental Evaluation

### Environment Installation

In our experiments, the operating system is Linux 4.9. We conduct experimets on MySQL 5.7.19.

1. Preparations: Python == 3.7

2. Install packages

   ```shell
   pip install -r requirements.txt
   pip install .
   ```

3. Download and install MySQL 5.7.19 and boost

   ```shell
   wget http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.gz
   wget https://dev.mysql.com/get/Downloads/MySQL-5.7/mysql-boost-5.7.19.tar.gz
   
   sudo cmake . -DCMAKE_INSTALL_PREFIX=PATH_TO_INSTALL -DMYSQL_DATADIR=PATH_TO_DATA -DDEFAULT_CHARSET=utf8 -DDEFAULT_COLLATION=utf8_general_ci -DMYSQL_TCP_PORT=3306 -DWITH_MYISAM_STORAGE_ENGINE=1 -DWITH_INNOBASE_STORAGE_ENGINE=1 -DWITH_ARCHIVE_STORAGE_ENGINE=1 -DWITH_BLACKHOLE_STORAGE_ENGINE=1 -DWITH_MEMORY_STORAGE_ENGINE=1 -DENABLE_DOWNLOADS=1 -DDOWNLOAD_BOOST=1 -DWITH_BOOST=PATH_TO_BOOST;
   sudo make -j 16;
   sudo make install;
   ```



### Workload Preparation 

#### SYSBENCH

Download and install

```shell
git clone https://github.com/akopytov/sysbench.git
./autogen.sh
./configure
make && make install
```

Load data

```shell
sysbench --db-driver=mysql --mysql-host=$HOST --mysql-socket=$SOCK --mysql-port=$MYSQL_PORT --mysql-user=root --mysql-password=$PASSWD --mysql-db=sbtest --table_size=800000 --tables=150 --events=0 --threads=32 oltp_read_write prepare > sysbench_prepare.out
```



#### OLTP-Bench

We install OLTP-Bench to use the following workload: TPC-C, SEATS, Smallbank, TATP, Voter, Twitter, SIBench.

- Download

```
git clone https://github.com/oltpbenchmark/oltpbench.git
```

- To run `oltpbenchmark` outside the folder, modify the following file:

  - ./src/com/oltpbenchmark/DBWorkload.java (Line 85)

    ```shell

    pluginConfig = new XMLConfiguration("PATH_TO_OLTPBENCH/config/plugin.xml"); # modify this

    ```

  - ./oltpbenchmark

    ```

    #!/bin/bash

    java -Xmx8G -cp `$OLTPBENCH_HOME/classpath.sh bin` -Dlog4j.configuration=$OLTPBENCH_HOME/log4j.properties com.oltpbenchmark.DBWorkload $@

    ```

  - ./classpath.sh

    ```shell

    #!/bin/bash

    echo -ne "$OLTPBENCH_HOME/build"

    for i in `ls $OLTPBENCH_HOME/lib/*.jar`; do

        # IMPORTANT: Make sure that we do not include hsqldb v1

        if [[ $i =~ .*hsqldb-1.* ]]; then

            continue

        fi

        echo -ne ":$i"

    done

    ```

- Install 

  ```shell
  ant bootstrap
  ant resolve
  ant build
  ```



#### Join-Order-Benchmark (JOB)

Download IMDB Data Set from http://homepages.cwi.nl/~boncz/job/imdb.tgz.

Follow the instructions of https://github.com/winkyao/join-order-benchmark to load data into MySQL.



### Environment Variables

Before running the experiments, the following environment variables require to be set.

```shell
export SYSBENCH_BIN=PATH_TO_sysbench/src/sysbench
export OLTPBENCH_BIN=PATH_TO_oltpbench/oltpbenchmark
export MYSQLD=PATH_TO_mysqlInstall/bin/mysqld
export MYSQL_SOCK=PATH_TO_mysql/base/mysql.sock
export MYCNF=PATH_TO_autotune/template/experiment_normandy.cnf
export DATADST=PATH_TO_mysql/data
export DATASRC=PATH_TO_mysql/data_copy
```



### Experiments Design

All optimization methods are listed as follows:

| Method                                         | String of ${METHOD} |
| ---------------------------------------------- | ------------------- |
| Vanilla BO                                     | VBO                 |
| Mixed-Kernel BO                                | MBO                 |
| Sequential Model-based Algorithm Configuration | SMAC                |
| Tree-structured Parzen Estimator               | TPE                 |
| Trust-Region BO                                | TURBO               |
| Deep Deterministic Policy Gradient             | DDPG                |
| Genetic Algorithm                              | GA                  |



#### Exp.1: Tuning improvement over knob set generated by different important measurements.

Compared importance measurements: `lasso`, `gini`, `fanova`, `ablation`, `shap`.

To conduct the experiment shown in Figure 3(a), the script is as follows. Please specify `${lhs_log}`.

```shell
python train.py --knobs_config=experiment/gen_knobs/JOB_lasso.json    --knobs_num=5 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_gini.json     --knobs_num=5 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_fanova.json   --knobs_num=5 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_ablation.json --knobs_num=5 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_shap.jso      --knobs_num=5 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}

python train.py --knobs_config=experiment/gen_knobs/JOB_lasso.json    --knobs_num=20 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_gini.json     --knobs_num=20 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_fanova.json   --knobs_num=20 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_ablation.json --knobs_num=20 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
python train.py --knobs_config=experiment/gen_knobs/JOB_shap.jso      --knobs_num=20 --method=VBO --workload=job --dbname=imdboload --y_variable=lat --lhs_num=10 --lhs_log=${lhs_log}
```

To conduct the experiments in (b), (c), and (d), modify `${knobs_num}`,`${method}`,`${workload}`, `${dbname}`, and `${y_variable}`, where

- `${knobs_num}` = 5, 20

- `${method}` = VBO, DDPG

- `${workload}` = job, sysbench

  - if `${workload} == job`, then `${dbname} = imdbload`, `${y_variable}=lat` 
  - if `${workload} == sysbench`, then `${dbname} =sbtest `, `${y_variable}=tps` 



Note`${knobs_config}` indicates the configuration file where knobs are ranked by importance.

- We provide the configuration file generated on our VM: `experiment/gen_knobs/${workload}_${measure}.json`. 
- You can also generate new configuration file with samples in your environment.

  
#### Exp.2: Performance improvement and tuning cost when increasing the number of tuned knobs.

To conduct the experiment shown in Figure 5 (a) and 5 (b), the script is as follows.

```shell
python train.py --method=VBO --workload=job --dbname=imdbload --y_variable=lat --lhs_num=10 --knobs_num=${knobs_num} --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=VBO --workload=sysbench --dbname=sbtest --y_variable=tps --lhs_num=10 --knobs_num=${knobs_num} --knobs_config=experiment/gen_knobs/SYSBENCH_shap.json --lhs_log=${lhs_log}
```

 Please specify `${knobs_num}` and `${lhs_log}`, where

- `${knobs_num} ` = 5, 10, 15, 20, 30, 50, 70, 90, 197 


  
#### Exp.3: Incremental Knob Selection.

Compared methods: `5 Knobs`, `20 Knobs`, `increase`, `decrease`.

To conduct the experiment shown in Figure 6(a), the script is as follows. Please specify `${lhs_log}`.

```shell
python train.py --method=VBO       --knobs_num=5  --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=VBO       --knobs_num=20 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=increase --knobs_num=-1 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=decrease   --knobs_num=-1 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
```

To conduct the experiment shown in (b), you can 

- replace `--workload=JOB --y_variable=lat` with `--workload=sysbench --y_variable=tps`


  
#### Exp.4: Optimizer comparision on different configuration space.

Compared optimizers: `VBO`, `MBO`, `SMAC`, `TPE`, `TURBO`, `DDPG`, `GA`.

To conduct the experiment shown in Figure 7(a), the script is as follows. Please specify `${lhs_log}`.

```shell
python train.py --method=VBO   --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=MBO   --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=SMAC  --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=TPE   --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=TURBO --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=DDPG  --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
python train.py --method=GA    --knobs_num=5 --workload=job --y_variable=lat --dbname=imdbload --knobs_config=experiment/gen_knobs/JOB_shap.json --lhs_log=${lhs_log}
```

To conduct the experiment shown in (b), (c), (d), (e), (f), and (g), you can 

- replace `--knobs_num=5` with`--knobs_num=20` or `--knobs_num=197`
- replace `--workload=JOB --y_variable=lat --dbname=imdbload` with `--workload=sysbench --y_variable=tps --dbname=sbtest`


  
#### Exp.5: Comparison experiment for knobs heterogeneity.

Compared optimizers: `VBO`, `MBO`, `SMAC`, `DDPG`.

To conduct the experiment shown in Figure 8(a) and (b), the script is as follows.

```shell
python train.py --method=${method} --knobs_num=20 --workload=job --y_variable=lat --dbname=${dbname}   --knobs_config=experiment/gen_knobs/JOB_continuous.json --lhs_log=${lhs_log} --lhs_num=10
python train.py --method=${method} --knobs_num=20 --workload=job --y_variable=lat --dbname=${dbname}   --knobs_config=experiment/gen_knobs/JOB_heterogeneous.json --lhs_log=${lhs_log} --lhs_num=10
```

 Please specify `${method}`, `${dbname}` and `${lhs_log}`, where

- `${method}` is one of `VBO`, `MBO`, `SMAC`, `DDPG`.

  
  
#### Exp.6: Algorithm overhead comparison.

Compared optimizers: `MBO`, `SMAC`, `TPE`, `TURBO`, `DDPG`, `GA`.

To conduct the experiment shown in Figure 8(a) and (b), the script is as follows.

```shell
python train.py --method=${method} --knobs_num=20 --workload=job --y_variable=lat --dbname=${dbname}   --knobs_config=experiment/gen_knobs/job_shap.json --lhs_log=${lhs_log} --lhs_num=10
```

 Please specify `${method}`, `${dbname}` and `${lhs_log}`, where

- `${method}` is one of  `MBO`, `SMAC`, `TPE`, `TURBO`, `DDPG`, `GA`.

Note if you have already done Exp.4, you can skip running the above script and analyze log files in `script/log/`.

  
  
#### Exp.7: Transfering methods comparison.

Compared methods: `RGPE-MBO`, `RGPE-SMAC`, `MAP-MBO`, `MAP-SMAC`, `FineTune-DDPG`

To conduct the experiment shown in Table 9, there are two steps: 

- Pre-train on source workloads (`Smallbank`, `SIBench`, `Voter`, `Seats`, `TATP`);
- Validate on target workloads (`TPCC`, `SYSBENCH`, `Twitter`).

Scripts for pre-trains is similar to the ones for Exp.4

To validate on target workloads, the scripts are as follows.

```
python train.py --method=MBO  --RGPE --source_repo=${repo}         --knobs_num=20 --workload=job --y_variable=lat --dbname=tpcc   --knobs_config=experiment/gen_knobs/oltp.json --lhs_log=${lhs_log} --lhs_num=10 
python train.py --method=SMAC --RGPE --source_repo=${repo}         --knobs_num=20 --workload=job --y_variable=lat --dbname=tpcc   --knobs_config=experiment/gen_knobs/oltp.json --lhs_log=${lhs_log} --lhs_num=10  
python train.py --method=MBO  --workload_map --source_repo=${repo} --knobs_num=20 --workload=job --y_variable=lat --dbname=tpcc   --knobs_config=experiment/gen_knobs/oltp.json --lhs_log=${lhs_log} --lhs_num=10 
python train.py --method=SMAC --workload_map --source_repo=${repo} --knobs_num=20 --workload=job --y_variable=lat --dbname=tpcc   --knobs_config=experiment/gen_knobs/oltp.json --lhs_log=${lhs_log} --lhs_num=10 
python train.py --method=DDPG --params=model_params/${ddpg_params} --knobs_num=20 --workload=job --y_variable=lat --dbname=tpcc   --knobs_config=experiment/gen_knobs/oltp.json --lhs_log=${lhs_log} --lhs_num=10 
```

Note that 

- for `RGPE-` methods, you should specify `--RGPE --source_repo=${repo} `
- for `MAP-` methods, you should specify `--workload_map --source_repo=${repo}`
- for `FineTune-DDPG`, you should specify `--params=model_params/${ddpg_params}`


  
## Project Code Overview

+ `autotune/tuner.py` : the implemented optimization methods.
+ `autotune/dbenv.py` : the interacting functions with database.
+ `script/train.py` : the python script to start an experiment.
+ `script/experiment/gen_knob` : the knob importance ranking files generated by different methods.
