#!/usr/bin/env python

import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from env.training_environment import Env as train_env
from env.testing_environment import Env as test_env

from ppo_alg import PPO_agent
from ddpg_alg import DDPG_agent
from hrl_alg import HRL_agent
from ok_alg import OK_agent
from ok_alg_cts import OK_agent_cts

import argparse

MAX_STEPS = 500
MAX_EPISODES = 200

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('env_type', type=str,
                        help='The environment, train vs test')
    parser.add_argument('agent_type', type=str,
                        help='The agent. PPO, DDPG, HRL, ...')
    parser.add_argument('module_index', type=int, default=None,
                        help='Which training dojo to use. [0-7)')
    parser.add_argument('save_path', type=str, default='/tmp/ros/test',
                        help='A directory to save the models')
    parser.add_argument('load_path', type=str, default=None,
                        help='A directory to load a model from')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:-2])
    if args.load_path == 'None':
        args.load_path = None
    print(args)

    rospy.init_node('run_agent')

    # Choose correct environment
    if (args.env_type == "train"):
        Env = train_env
    elif (args.env_type == "test"):
        Env = test_env
    else:
        raise ValueError('need a valid environment')

    # sort out saving / load directories
    sys.path.append(args.load_path)
    sys.path.append(args.save_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('Saving in: {}'.format(args.save_path))

    # init agent
    if (args.agent_type == "ppo"):
        env = Env("PPO", args.module_index)
        agent = PPO_agent(args.load_path, env, MAX_STEPS, args.save_path)
    elif (args.agent_type == "ddpg"):
        env = Env("DDPG", args.module_index)
        agent = DDPG_agent(0, env, MAX_STEPS)
    elif (args.agent_type == "hrl") and (args.env_type == "test"):
        env = Env("HRL")
        agent = HRL_agent(args.load_path, env, MAX_STEPS, args.save_path, 7)
    elif (args.agent_type == "ok") and (args.env_type == "test"):
        env = Env("ok")
        agent = OK_agent(args.load_path, env, MAX_STEPS, args.save_path, 7)
    elif (args.agent_type == "okcts") and (args.env_type == "test"):
        env = Env("okcts")
        agent = OK_agent_cts(args.load_path, env, MAX_STEPS, args.save_path, 7)

    else:
        raise ValueError('enter valid args...')

    # play the game...
    for ep in range(MAX_EPISODES):
        collision = 0
        goal = 0
        running_reward = 0
        ep_steps = 0
        state = env.reset()

        for step in range(MAX_STEPS):
            ep_steps += 1
            state, reward, collision, goal = agent.step(state, ep)
            running_reward += reward
            if (collision or goal or step == MAX_STEPS - 1):
                break

        env.logEpisode(running_reward, collision, goal, ep_steps)
        print("Episode " + str(ep))

    agent.save(ep)

    print("Max episodes reached")
