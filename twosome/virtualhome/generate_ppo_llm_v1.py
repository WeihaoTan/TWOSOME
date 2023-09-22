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

from transformers import GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

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

    # print("play virtual home v1")
    # env = gym.make("VirtualHome-v1", debug=True)
    # obs = env.reset()
    # print(obs)
    # text, action_list = env.obs2text(obs)
    # print(action_list)
    # print("\""+text+"\",")
    # print('-------------')
    #
    # actions = [4, 6, 5, 8, 7, 9]
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

    # env_params = {
    #     'seed': 10,
    #     'debug': False,
    # }

    # print("play virtual home v1")

    # envs = gym.vector.SyncVectorEnv(
    #     [make_env("VirtualHome-v1", 10, 0, False, "tmp", env_params) for i in
    #      range(1)]
    # )

    # print("play agent")
    agent = LLMAgent(normalization_mode="word",
                     load_path=os.path.join(root, "workdir", "VirtualHome-v1__heat_pancake_ppo_llm__10__20230913_05_53_46", "saved_models", "epoch_0049"),
                     load_8bit=False)

    # obs = envs.reset()
    # inference(obs, agent, device)
    # print("-" * 100)

    # actions = [4, 6, 5, 8, 7, 9]

    # for i in actions:
    #     obs, reward, done, info = envs.step(np.array([i]))
    #     inference(obs, agent, device)
    #     print("-" * 100)
    
    #llama = agent.llama
    
    twosome = agent.actor
    tokenizer = agent.tokenizer


    # generation_config = GenerationConfig(
    #     temperature=0.9,
    #     top_p=0.75,
    #     num_beams=4,
    # )

    prompt = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the kitchen. You notice pancake and microwave. Currently, you are not grabbing anything in hand. The pancake and the microwave are not within your immediate reach. The microwave is not opend. In order to heat up the pancake in the microwave, your next step is to"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    twosome_ids = twosome.generate(
        input_ids=input_ids,
        #generation_config=generation_config,
        max_new_tokens=30
    )

    twosome_output = tokenizer.batch_decode(twosome_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("TWOSOME")
    print(twosome_output)
    print("*" * 100)

    with open("twosome.txt", "w") as f:
        f.write(str(twosome))
        
        
    #llama = agent.actor.get_base_model()
    llama = agent.llama
    llama_ids = llama.generate(
        input_ids=input_ids,
        #generation_config=generation_config,
        max_new_tokens=30
    )
    llama_output = tokenizer.batch_decode(llama_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("LLaMA")
    print(llama_output)
    print("*" * 100)

    with twosome.disable_adapter():
        with open("llama.txt", "w") as f:
            f.write(str(twosome))

    ori_llama = LlamaForCausalLM.from_pretrained(
        'decapoda-research/llama-7b-hf',
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
        cache_dir=os.path.join(root, 'weights/llama')
    )
    llama_ids = ori_llama.generate(
        input_ids=input_ids,
        #generation_config=generation_config,
        max_new_tokens=30
    )
    llama_output = tokenizer.batch_decode(llama_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("LLaMA")
    print(llama_output)

    with open("ori_llama.txt", "w") as f:
        f.write(str(ori_llama))
    
    base = agent.actor.get_base_model()
    with open("actor_base.txt", "w") as f:
        f.write(str(base))

    # for s in generation_output.sequences:
    #     output = tokenizer.decode(s)
    #     pprint("Resposta: " + output.split("### Resposta:")[1].strip())


if __name__ == '__main__':
    main()

