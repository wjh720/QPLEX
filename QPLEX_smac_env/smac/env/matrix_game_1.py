from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class Matrix_game1Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=2,
            n_actions=100,
            reward_win=10,
            first_weight=10.4,
            episode_limit=50,
            coor_len=3,
            coor_list=None,
            obs_last_action=True,
            state_last_action=True,
            is_print=False,
            print_rew=False,
            print_steps=1000,
            seed=None
    ):
        # Map arguments
        self.print_rew = print_rew
        self.is_print = is_print
        self.print_steps = print_steps
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents

        # Rewards args
        self.reward_win = reward_win

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 3

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False

        self.episode_limit = 1

        self.matrix_table = np.array([[first_weight, 0, 10], [0, 10, 10], [10, 10, 10]])

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = self.matrix_table[actions[0]][actions[1]]

        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return np.array([self._episode_steps])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 1

    def get_state(self):
        """Returns the global state."""
        return np.array([self._episode_steps])

    def get_state_size(self):
        """Returns the size of the global state."""
        return 1

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
