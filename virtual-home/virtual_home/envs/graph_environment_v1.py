from .base_environment import BaseEnvironment
# from utils import utils_environment as utils

import sys
import os

from gym import Wrapper, spaces




curr_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f'{curr_dir}/../../virtualhome/')
sys.path.append(f'{curr_dir}/../../vh_mdp/')

from virtual_home.envs.graph_env_v1 import VhGraphEnv
import pdb
import random
import numpy as np
import copy

import os
current_path = os.path.dirname(os.path.realpath(__file__))

class GraphEnvironment(Wrapper):
    def __init__(self,
                 state_path = os.path.join(current_path, 'turn_on_tv.json'),
                 num_agents=1,
                 max_episode_length=50,
                 observation_types=None,
                 output_folder=None,
                 debug = False,
                 seed=123):

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.debug = debug
        self.steps = 0
        self.max_episode_length = max_episode_length
        #self.output_folder = output_folder
        self.env = VhGraphEnv(state_path = state_path, n_chars=num_agents)



        self.obs_keys = [
                         'in_kitchen',
                         'in_bathroom',
                         'in_bedroom',
                         'in_livingroom',

                         'see_pancake',
                         'close_to_pancake',
                         'hold_pancake',

                         'see_microwave',
                         'close_to_microwave',
                         'is_microwave_open',

                         'pancake_in_microwave',
                    ]

        self.rooms = ['kitchen', 'livingroom', 'bathroom', 'bedroom']
        self.items = ['pancake', 'microwave']

        self.create_action_list()
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.obs_keys),), dtype=np.float32)

    def reset(self):
        self.env.reset()
        self.env.to_pomdp()
        self.steps = 0
        self.total_return = 0
        self.discount = 1
        vector_obs, dict_obs = self.get_vector_obs(self.env.get_observations())
        return dict_obs if self.debug else vector_obs

    def step(self, action):
        action_dict = {0: self.action_list[action]}
        obs, r, d, info = self.env.step(action_dict)

        self.steps += 1
        vector_obs, dict_obs = self.get_vector_obs(obs[0])
        obs = dict_obs if self.debug else vector_obs

        reward, done = self.reward(dict_obs)

        self.total_return += self.discount * reward
        self.discount *= 0.95 # 0.99

        if done:
            episode_info = {
                "r": self.total_return,
                "l": self.steps,
            }
            info["episode"] = episode_info    

        return obs, reward, done, info

    def reward(self, dict_obs):
        reward = 0
        done = False

        if dict_obs["in_kitchen"] == 1 and dict_obs["see_pancake"] == 0:
            reward += 1.0
            done = True

        if self.steps >= self.max_episode_length:
            done = True 
        return reward, done
    
    def get_vector_obs(self, obs):
        obs_dict = {k: 0 for k in self.obs_keys}

        for node in obs['nodes']:
            for item in self.items:
                if node['id'] == self.env.target_id[item]:
                    obs_dict['see_' + item] = 1
                    if item == 'microwave' and 'OPEN' in node['states']:
                        obs_dict['is_microwave_open'] = 1

        for edge in obs['edges']:
            if edge['from_id'] == self.env.target_id['character']:
                for room in self.rooms:
                    if self.env.target_id[room] == edge['to_id'] and edge['relation_type'] == 'INSIDE':
                        obs_dict['in_' + room] = 1
            
                for item in self.items:
                    if self.env.target_id[item] == edge['to_id']:
                        if edge['relation_type'] == 'CLOSE':
                            obs_dict['close_to_' + item] = 1
                    
                        elif edge['relation_type'] in ['HOLDS_LH', 'HOLDS_RH'] and item in ['pancake']:
                            obs_dict['hold_' + item] = 1
                        
            elif edge['from_id'] == self.env.target_id['pancake']:
                if edge['relation_type'] == 'INSIDE' and edge['to_id'] == self.env.target_id['microwave']:
                    obs_dict['pancake_in_microwave'] = 1

        return [value for value in obs_dict.values()], obs_dict
    
    def create_action_list(self):
        self.action_list = [
            '[WALK] <livingroom> ({})'.format(self.env.target_id['livingroom']),
            '[WALK] <kitchen> ({})'.format(self.env.target_id['kitchen']),
            '[WALK] <bathroom> ({})'.format(self.env.target_id['bathroom']),
            '[WALK] <bedroom> ({})'.format(self.env.target_id['bedroom']),
            
            '[WALK] <pancake> ({})'.format(self.env.target_id['pancake']),
            '[WALK] <microwave> ({})'.format(self.env.target_id['microwave']),

            '[GRAB] <pancake> ({})'.format(self.env.target_id['pancake']),
            '[PUTIN] <pancake> ({}) <microwave> ({})'.format(self.env.target_id['pancake'], self.env.target_id['microwave']),

            '[OPEN] <microwave> ({})'.format(self.env.target_id['microwave']),
            '[CLOSE] <microwave> ({})'.format(self.env.target_id['microwave']),
        ]

        self.action_template = [
            "walk to the living room", # 0
            "walk to the kitchen", # 1
            "walk to the bathroom", # 2
            "walk to the bedroom", # 3

            "walk to the pancake", # 4
            "walk to the microwave", # 5

            "grab the pancake", # 6

            "put the pancake in the microwave",  # 7

            'open the microwave', # 8
            'close the microwave', # 9
        ]

    def obs2text(self, obs):
        """
        {'in_kitchen': 1,
        'in_liveingroom': 0,
        'in_bathroom': 0,
        'in_bedroom': 0,
        'see_pancake': 1,
        'close_to_pancake': 0,
        'hold_pancake': 0,
        'see_microwave': 1,
        'close_to_microwave': 0,
        'is_microwave_open': 0,
        'pancake_in_microwave': 0
        }

         'in_kitchen',
         'in_bathroom',
         'in_bedroom',
         'in_liveingroom',

         'see_pancake',
         'close_to_pancake',
         'hold_pancake',

         'see_microwave',
         'close_to_microwave',
         'is_microwave_open',

         'pancake_in_microwave',
        """

        text = ""

        if self.debug:
            in_kitchen = int(obs['in_kitchen'])
            in_bathroom = int(obs['in_bathroom'])
            in_bedroom = int(obs['in_bedroom'])
            in_livingroom = int(obs['in_livingroom'])

            see_pancake = int(obs['see_pancake'])
            close_to_pancake = int(obs['close_to_pancake'])
            hold_pancake = int(obs['hold_pancake'])

            see_microwave = int(obs['see_microwave'])
            close_to_microwave = int(obs['close_to_microwave'])
            is_microwave_open = int(obs['is_microwave_open'])

            pancake_in_microwave = int(obs['pancake_in_microwave'])

        else:
            in_kitchen = obs[0]
            in_bathroom = obs[1]
            in_bedroom = obs[2]
            in_livingroom = obs[3]

            see_pancake = obs[4]
            close_to_pancake = obs[5]
            hold_pancake = obs[6]

            see_microwave = obs[7]
            close_to_microwave = obs[8]
            is_microwave_open = obs[9]

            pancake_in_microwave = obs[10]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        object_text = ""
        action_list = []

        if in_kitchen:

            if not see_pancake:
                object_text += "The pancake is in the microwave. "
            else:
                object_text += "You notice pancake and microwave. "

            if hold_pancake:
                object_text += "Currently, you have grabbed the pancake in hand. "
                if close_to_microwave:
                    object_text += "The microwave is close to you. "
                    action_list = [0,2,3,4,7,8,9]
                else:
                    object_text += "The microwave is not close to you. "
                    action_list = [0,2,3,4,5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is close to you. "
                    action_list = [0,2,3,5,6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                    action_list = [0,2,3,4,8,9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not close to you. "
                    action_list = [0,2,3,4,5]
                else:
                    if is_microwave_open:
                        action_list = [0,2,3,8,9]
                    else:
                        action_list = [0, 2, 3,9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0,2,3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0,1,3]
        elif in_bedroom:

                if hold_pancake:
                    object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
                else:
                    object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

                action_list = [0,1,2]
        elif in_livingroom:

                if hold_pancake:
                    object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
                else:
                    object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

                action_list = [1,2,3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        action_template = [self.action_template[i] for i in action_list]

        return text, action_template