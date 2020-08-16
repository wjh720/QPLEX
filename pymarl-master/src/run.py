import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np
import copy as cp
import random

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          args.burn_in_period,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3' \
            or args.env == 'mmdp_game_1':
        last_demo_T = -args.demo_interval - 1

    while runner.t_env <= args.t_max:

        if not args.is_batch_rl:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        for _ in range(args.num_circle):
            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                if args.is_batch_rl:
                    runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

                if args.env == 'mmdp_game_1' and args.learner == "q_learner_exp":
                    for i in range(int(learner.target_gap) - 1):
                        episode_sample = buffer.sample(args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)

                        learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 :

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
        if args.env == 'mmdp_game_1' and \
                (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
            ### demo
            episode_sample = cp.deepcopy(buffer.sample(1))
            for i in range(args.n_actions):
                for j in range(args.n_actions):
                    new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                    if i == 0 and j == 0:
                        rew = th.Tensor([1, ])
                    else:
                        rew = th.Tensor([0, ])
                    if i == 1 and j == 1:
                        new_obs = th.Tensor([1, 0]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                    else:
                        new_obs = th.Tensor([0, 1]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    episode_sample['actions'][0, :, :, 0] = new_actions
                    episode_sample['obs'][0, 1:, :, :] = new_obs
                    episode_sample['reward'][0, 0, 0] = rew
                    new_actions_onehot = th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,))
                    new_actions_onehot = new_actions_onehot.scatter_(3, episode_sample['actions'].cpu(), 1)
                    episode_sample['actions_onehot'][:] = new_actions_onehot

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    #print("action pair: %d, %d" % (i, j))
                    learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
            last_demo_T = runner.t_env
            #time.sleep(1)

        if (args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3') and \
                (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
            ### demo
            episode_sample = cp.deepcopy(buffer.sample(1))
            for i in range(args.n_actions):
                for j in range(args.n_actions):
                    new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    episode_sample['actions'][0, :, :, 0] = new_actions
                    new_actions_onehot = th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,)).cuda()
                    new_actions_onehot = new_actions_onehot.scatter_(3, episode_sample['actions'].cuda(), 1)
                    episode_sample['actions_onehot'][:] = new_actions_onehot
                    if i == 0 and j == 0:
                        rew = th.Tensor([8, ])
                    elif i == 0 or j == 0:
                        rew = th.Tensor([-12, ])
                    else:
                        rew = th.Tensor([0, ])
                    if args.env == 'matrix_game_3':
                        if i == 1 and j == 1 or i == 2 and j == 2:
                            rew = th.Tensor([6, ])
                    episode_sample['reward'][0, 0, 0] = rew

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    #print("action pair: %d, %d" % (i, j))
                    learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
            last_demo_T = runner.t_env
            #time.sleep(1)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            if args.double_q:
                os.makedirs(save_path + '_x', exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run * args.num_circle

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if args.is_save_buffer and save_buffer.is_from_start:
        save_buffer.is_from_start = False
        save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
