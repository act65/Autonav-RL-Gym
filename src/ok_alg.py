#!/usr/bin/env python


import rospy
import os
import numpy as np
import gc
import time
import sys
import itertools
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

class OK_agent:
    def __init__(self, load_path, env, max_timesteps, save_path, n_policies, T=3):
        self.T = T
        self.memory = Memory()
        self.ppo = PPO(state_dim, n_policies, hidden_dim, lr, betas, gamma,
                       K_epochs, eps_clip, save_path, load_path)

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
        self.past_action = np.array([0., .0])
        self.max_timesteps = max_timesteps
        self.env_actions = [[0,0], [0,ACTION_V_MAX], [ACTION_V_MAX, 0], [ACTION_V_MAX, ACTION_V_MAX]]
        self.weight_actions = [np.reshape(np.array(i), n_policies) for i in itertools.product([-1, 0, 1], repeat = n_policies)]
        # print(len(self.weight_actions)) = 2187

        # advantage over HRL with options over pretrained policies
        # can use combinations of the policies

    def step(self, state, *args):
        self.time_step += 1
        a = self.ppo.select_action(state, self.memory)
        state, reward, collision, goal = self.option_keyboard(state, self.weight_actions[a])
        self.memory.rewards.append(reward)
        self.memory.masks.append(float(collision or self.time_step == self.max_timesteps - 1))

        if (self.time_step % (update_timestep) == 0):
            self.ppo.update(self.memory)
            self.memory.clear_memory()
            self.time_step = 0

        return state, reward, collision, goal

    def option_keyboard(self, state, weights):
        total_reward = 0
        total_collision = False
        terminated = False
        steps = 0
        goal = 0

        while not terminated and not total_collision:
            steps += 1
            # how to augment these with termination actions!? could;
            # - fixed large R at T timesteps!?
            # - linearly increase the R for terminating???
            values = np.stack([policy.value(state) for policy in self.policies], axis=1)
            weighted_values = np.dot(values, weights)  # (n_actions, n_fns)  x n_fns = n_actions
            a = sample(weighted_values)
            action = self.env_actions[a]

            if weighted_values[a] > 0.01*steps:
                # if not terminate option, then exectute action
                state, reward, collision, goal = self.env.step(action, self.past_action)
                self.past_action = action

                total_reward += reward
                total_collision = total_collision or collision
            else:
                terminated = True

        return state, total_reward, total_collision, goal

    def save(self, ep):
        self.ppo.save_models(ep)


def softmax(x, axis=-1):
    return np.exp(x)/np.sum(np.exp(x), keepdims=True, axis=axis)

def sample(x, axis=-1):
    p = softmax(x)
    g = -np.log(-np.log(np.random.random(x.shape)))
    idx = np.argmax(np.log(p) + g, axis=axis)
    return idx
