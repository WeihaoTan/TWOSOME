import sys
import os
import json
import gym
import virtual_home

import time
if __name__ == '__main__':

    env = gym.make("VirtualHome-v1", debug=True)
    obs = env.reset()
    print(obs)
    text, action_list = env.obs2text(obs)
    print(action_list)
    print("\""+text+"\",")
    print('-------------')

    actions = [4, 6, 5, 8, 7, 9]

    for i in actions:
        obs, reward, done, info = env.step(i)
        print(env.action_list[i], info, reward, done)
        print(obs)
        text, action_list = env.obs2text(obs)
        print(action_list)
        print("\""+text+"\",")
        print('-------------')

    env = gym.make("VirtualHome-v2", debug=True)
    obs = env.reset()
    print(obs)
    text, action_list = env.obs2text(obs)
    print(action_list)
    print("\""+text+"\",")
    print('-------------')

    actions = [4, 9, 5, 10, 0, 6, 11, 7, 13, 8, 15]

    for i in actions:
        obs, reward, done, info = env.step(i)
        print(env.action_list[i], info, reward, done)
        print(obs)
        text, action_list = env.obs2text(obs)
        print(action_list)
        print("\""+text+"\",")
        print('-------------')
    print(info)
