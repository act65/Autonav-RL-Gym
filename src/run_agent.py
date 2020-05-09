#!/usr/bin/env python

import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
from collections import deque
from ppo.storage import Memory
from ppo.ppo_models import PPO
from env.training_environment import Env as train_env
from env.testing_environment import Env as test_env
from ppo_alg import PPO_agent
from ddpg_alg import DDPG_agent
from hrl_alg import HRL_agent

MAX_STEPS = 500
MAX_EPISODES = 200

if __name__ == '__main__':

    rospy.init_node('run_agent')

    load_ep = 0

    # Choose correct environment
    if (sys.argv[1] == "train"):
        Env = train_env
    elif (sys.argv[1] == "test"):
        Env = test_env
    else:
        raise ValueError('need a valid environment')

    # arg 3 set load ep if specified
    if len(sys.argv) <= 4 + 2:
        load_ep = 0
    else:
        load_ep = int(sys.argv[4])

    module_index = int(sys.argv[3])

    #---Directory Path---#
    dirPath = "/home/act65/catkin_ws/src/Autonav-RL-Gym/saved_models/ppo-{}".format(module_index)
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    sys.path.append(dirPath)
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    print('Saving in: {}'.format(dirPath))

    # arg 2, agent PPO, DDPG, initialize Env(PPO) etc
    if (sys.argv[2] == "ppo"):
        env = Env("PPO", module_index)
        agent = PPO_agent(load_ep, env, MAX_STEPS, dirPath)
    elif (sys.argv[2] == "ddpg"):
        env = Env("DDPG", module_index)
        agent = DDPG_agent(load_ep, env, MAX_STEPS)
    elif (sys.argv[2] == "hrl") and (sys.argv[1] == "test"):
        env = Env("HRL")
        agent = HRL_agent(load_ep, env, MAX_STEPS, dirPath, 7)
    else:
        raise ValueError('enter valid args...')


    for ep in range(load_ep, MAX_EPISODES):
        collision = 0
        goal = 0
        running_reward = 0
        ep_steps = 0
        state = env.reset()

        for step in range(MAX_STEPS):
            ep_steps += 1
            state, reward, collision, goal = agent.step(state,ep)
            running_reward += reward
            if (collision or goal or step == MAX_STEPS - 1):
                break

        env.logEpisode(running_reward, collision, goal, ep_steps)
        print("Episode " + str(ep))

    if (sys.argv[1] == "train"):
        agent.save(ep)

    print("Max episodes reached")
