# QPLEX: Duplex Dueling Multi-Agent Q-Learning

## Note
 This codebase accompanies paper *Duplex Dueling Multi-Agent Q-Learning*, 
 and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced. The modified SMAC of QPLEX is illustrated in the folder `QPLEX_smac_env` of supplymentary material.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement learning](https://arxiv.org/abs/1905.05408)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

The following command train NDQ on the didactic task `matrix_game_2 `.

```shell
python3 src/main.py 
--config=qplex 
--env-config=matrix_game_2 
with 
local_results_path='../../../tmp_DD/sc2_bane_vs_bane/results/' 
save_model=True use_tensorboard=True 
save_model_interval=200000 
t_max=210000 
epsilon_finish=1.0
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To train QPLEX on SC2 offline setting tasks, run the following command:

Construct the dataset:

```shell
python3 src/main.py 
--config=qmix 
--env-config=sc2 
with 
env_args.map_name=1c3s5z 
env_args.seed=1 
local_results_path='../../../tmp_DD/sc2_1c3s5z/results/' 
save_model=True 
use_tensorboard=True 
save_model_interval=200000 
t_max=2100000 
is_save_buffer=True 
save_buffer_size=20000 
save_buffer_id=0
```

Training with offline data collection:

```shell
python3 src/main.py 
--config=qplex_sc2 
--env-config=sc2 
with 
env_args.map_name=1c3s5z 
env_args.seed=1 
local_results_path='../../../tmp_DD/sc2_1c3s5z/results/' 
save_model=True 
use_tensorboard=True 
save_model_interval=200000 
t_max=2100000 
is_batch_rl=True 
load_buffer_id=0
```

To train QPLEX on SC2 online setting tasks, run the following command:

```shell
python3 src/main.py 
--config=qplex_qatten_sc2 
--env-config=sc2 
with 
env_args.map_name=3s5z 
env_args.seed=1 
local_results_path='../../../tmp_DD/sc2_3s5z/results/' 
save_model=True 
use_tensorboard=True 
save_model_interval=200000 
t_max=2100000 
num_circle=2
```

SMAC maps can be found in in the folder `QPLEX_smac_env` of supplymentary material.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.