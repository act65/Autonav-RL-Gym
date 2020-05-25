#!/usr/bin/env python


import rospy
import os
import numpy as np
import gc
import time
import sys
from ppo.storage import Memory
from std_msgs.msg import Float32
# from env.training_environment import Env
from ppo.ppo_models import PPO

import torch

# hyperparams
update_timestep = 500      # update policy every n timesteps
hidden_dim = 256            # constant std for action distribution (Multivariate Normal)
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 2e-3                # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None

# state params
state_dim = 28
action_dim = 4

ACTION_V_MIN = 0  # m/s
ACTION_V_MAX = 0.4  # m/s

class HRL_agent:
    def __init__(self, load_path, env, max_timesteps, save_path, n_policies, T=3):
        self.T = T
        self.memory = Memory()
        self.ppo = PPO(state_dim, n_policies, hidden_dim, lr, betas, gamma,
                       K_epochs, eps_clip, save_path, None)

        if load_path is None:
            self.policies = [PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma,
                           K_epochs, eps_clip, save_path,
                           None)
                           for _ in range(n_policies)]
        else:
            self.policies = [PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma,
                           K_epochs, eps_clip, save_path,
                           os.path.join(load_path, 'ppo-{}'.format(i)))
                           for i in range(n_policies)]


        self.env = env
        self.time_step = 0
        self.past_action = np.array([0., .0])  # does the shape of this matter!?
        self.max_timesteps = max_timesteps
        self.actions = [[0,0], [0,ACTION_V_MAX], [ACTION_V_MAX, 0], [ACTION_V_MAX, ACTION_V_MAX]]

    def step(self, state, *args):
        self.time_step += 1
        option = self.ppo.select_action(state, self.memory)
        state, reward, collision, goal = self.multi_step(state, self.policies[option])
        self.memory.rewards.append(reward)
        self.memory.masks.append(float(collision or self.time_step == self.max_timesteps - 1))

        if (self.time_step % (update_timestep) == 0):
            self.ppo.update(self.memory)
            self.memory.clear_memory()
            self.time_step = 0

        return state, reward, collision, goal

    def multi_step(self, state, option):
        total_reward = 0
        total_collisions = False
        # what is the goal
        for _ in range(self.T):
            action = self.actions[option.select_action(state)]
            state, reward, collision, goal = self.env.step(action, self.past_action)
            self.past_action = action

            total_reward += reward
            total_collisions = total_collisions or collision

        print(goal)

        return state, total_reward, total_collisions, goal

    def save(self, ep):
        self.ppo.save_models(ep)
