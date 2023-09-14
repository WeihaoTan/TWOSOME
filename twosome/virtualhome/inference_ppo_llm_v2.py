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


def inference(obs, agent, device):
    obs = torch.Tensor(obs).to(device)
    text_obs = [agent.obs2text(o) for o in obs]
    prompt = [o["prompt"] for o in text_obs]

    action_list = [o["action"] for o in text_obs]

    sequence = []
    for p, ac in zip(prompt, action_list):
        sequence += [p + " " + a for a in ac]


    print("sequence", sequence)

    inputs = agent.tokenizer(sequence, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(agent.device)

    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = agent.actor(input_ids, attention_mask=attention_mask)

    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    for input_sentence, input_probs in zip(input_ids, gen_probs):
        for token, p in zip(input_sentence, input_probs):
            if token not in agent.tokenizer.all_special_ids:
                print(f"| {token:5d} | {agent.tokenizer.decode(token):8s} | {p.item():.3f} | {np.exp(p.item()):.2%}")

def main():

    # print("play virtual home v2")
    # env = gym.make("VirtualHome-v2", debug=True)
    # obs = env.reset()
    # print(obs)
    # text, action_list = env.obs2text(obs)
    # print(action_list)
    # print("\""+text+"\",")
    # print('-------------')
    #
    # actions = [4, 9, 5, 10, 0, 6, 11, 7, 13, 8, 15]
    #
    # for i in actions:
    #     obs, reward, done, info = env.step(i)
    #     print(env.action_list[i], info, reward, done)
    #     print(obs)
    #     text, action_list = env.obs2text(obs)
    #     print(action_list)
    #     print("\""+text+"\",")
    #     print('-------------')

    device = torch.device("cuda")

    env_params = {
        'seed': 100,
        'debug': False,
    }

    print("play virtual home v2")

    envs = gym.vector.SyncVectorEnv(
        [make_env("VirtualHome-v2", 100, 0, False, "tmp", env_params) for i in
         range(1)]
    )

    print("play agent")
    agent = LLMAgent(normalization_mode="word",
                     load_path=os.path.join(root, "workdir", "VirtualHome-v2__watch_tv_ppo_llm__100__20230913_08_39_20", "saved_models", "epoch_0199"),
                     load_8bit=False)

    obs = envs.reset()
    inference(obs, agent, device)
    print("-" * 100)

    actions = [4, 9, 5, 10, 0, 6, 11, 7, 13, 8, 15]

    for i in actions:
        obs, reward, done, info = envs.step(np.array([i]))
        inference(obs, agent, device)
        print("-" * 100)

if __name__ == '__main__':
    main()

