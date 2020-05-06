#!/usr/bin/env python


import rospy
import os
import numpy as np
import gc
import time
import sys
from ppo.storage import Memory
from std_msgs.msg import Float32
from env.training_environment import Env
from ppo.ppo_models import PPO

import torch

#---Directory Path---#
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))


# hyperparams
update_timestep = 500      # update policy every n timesteps
hidden_dim = 256            # constant std for action distribution (Multivariate Normal)
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 2e-4                # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None

# state params
state_dim = 28
action_dim = 4
ACTION_V_MIN = 0  # m/s
ACTION_V_MAX = 0.4  # m/s

memory = Memory()
ppo = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, ACTION_V_MIN, ACTION_V_MAX, eps_clip, dirPath)

class PPO_agent:

    def __init__(self, load_ep, env, max_timesteps):
        self.env = env
        self.time_step = 0
        self.past_action = np.array([0., 0.])
        self.max_timesteps = max_timesteps


        if (load_ep > 0):
            ppo.load_models(load_ep)


    # called every step
    def step(self, state, ep):

        self.time_step += 1
        action = ppo.select_action(state, memory)
        state, reward, collision, goal = self.env.step(action, self.past_action)

        self.past_action = action
        memory.rewards.append(reward)
        memory.masks.append(float(collision or self.time_step == self.max_timesteps - 1))

        if (self.time_step % update_timestep == 0):
            ppo.update(memory)
            memory.clear_memory()
            self.time_step = 0

        return state, reward, collision, goal

    def save(self, ep):
        ppo.save_models(ep)
