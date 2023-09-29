import os
import sys
import pathlib

root = str(pathlib.Path(__file__).parents[2])
sys.path.append(root)

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import virtual_home
from policy_v1 import LLMAgent

def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk

def main():

    device = torch.device("cuda")

    env_params = {
        'seed': 10,
        'debug': False,
    }

    print("play virtual home v1")

    envs = gym.vector.SyncVectorEnv(
        [make_env("VirtualHome-v1", 10, 0, False, "tmp", env_params) for i in
         range(1)]
    )

    print("play agent")
    load_path = os.path.join(root, "checkpoints", "food_preparation", "lora")


    agent = LLMAgent(normalization_mode="word",
                     load_path=load_path,
                     load_8bit=False)

    success_rate = 0

    reward_list = []
    step_list = []
    for i in range(100):
        steps = 0
        done = False
        rewards = 0
        discount = 1
        obs = envs.reset()
        while not done:
            steps += 1
            action = agent.get_action_and_value(obs, return_value= False)[0].cpu().numpy()
            print("action", action)
            obs, reward, done, info = envs.step(action)
            rewards += reward * discount
            discount *= 0.95
        reward_list.append(rewards)
        step_list.append(steps)
        if rewards > 0:
            success_rate += 1
    
        print(steps, rewards)

    print(np.mean(reward_list), np.std(reward_list))
    print(np.mean(step_list), np.std(step_list))
    print(success_rate)


if __name__ == '__main__':
    main()

