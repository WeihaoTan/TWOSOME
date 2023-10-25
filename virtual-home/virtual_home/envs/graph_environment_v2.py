from .base_environment import BaseEnvironment
# from utils import utils_environment as utils

import sys
import os

from gym import Wrapper, spaces

curr_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f'{curr_dir}/../../virtualhome/')
sys.path.append(f'{curr_dir}/../../vh_mdp/')

from virtual_home.envs.graph_env_v2 import VhGraphEnv
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



        self.obs_keys = ['in_kitchen', 'in_bathroom', 'in_bedroom', 'in_livingroom', 
                    'see_chips', 'close_to_chips', 'hold_chips', 'chips_on_coffeetable',
                    'see_milk', 'close_to_milk', 'hold_milk', 'milk_on_coffeetable',
                    'see_tv', 'close_to_tv', 'is_face_tv', 'is_tv_on',
                    'see_sofa', 'close_to_sofa', 'is_sit_sofa',
                    'see_coffeetable', 'close_to_coffeetable', 
                    ]

        self.rooms = ['kitchen', 'bathroom', 'bedroom', 'livingroom']
        self.items = ['chips', 'milk', 'tv', 'sofa', 'coffeetable']

        self.create_action_list()
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.obs_keys),), dtype=np.float32)

        self.first_grab_chips_flag = True
        self.first_grab_milk_flag = True
        self.pre_tv_state = None

    def reset(self):
        self.env.reset()
        self.env.to_pomdp()
        self.steps = 0
        self.total_return = 0
        self.discount = 1
        self.first_grab_chips_flag = True
        self.first_grab_milk_flag = True
        self.pre_tv_state = 0
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
        self.discount *= 0.95

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

        # if dict_obs["see_chips"] and dict_obs["close_to_chips"] and dict_obs["hold_chips"] and self.first_grab_chips_flag:
        #     reward += 0.1
        #     self.first_grab_chips_flag = False
        #
        # if dict_obs["see_milk"] and dict_obs["close_to_milk"] and dict_obs["hold_milk"] and self.first_grab_milk_flag:
        #     reward += 0.1
        #     self.first_grab_milk_flag = False
        #
        # if dict_obs["close_to_tv"] == 1 and self.pre_tv_state == 0 and dict_obs["is_tv_on"] == 1:
        #     reward += 0.1
        # if dict_obs["close_to_tv"] == 1 and self.pre_tv_state == 1 and dict_obs["is_tv_on"] == 0:
        #     reward -= 0.1
        #
        # self.pre_tv_state = dict_obs["is_tv_on"]

        if dict_obs['in_livingroom'] and dict_obs['is_face_tv'] and dict_obs['is_tv_on'] and dict_obs['is_sit_sofa'] \
            and (dict_obs['close_to_milk'] or dict_obs['milk_on_coffeetable']) \
            and (dict_obs['close_to_chips'] or dict_obs['chips_on_coffeetable']):
            reward += 1
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
                    if item == 'tv' and 'ON' in node['states']:
                        obs_dict['is_tv_on'] = 1

        for edge in obs['edges']:
            if edge['from_id'] == self.env.target_id['character']:
                for room in self.rooms:
                    if self.env.target_id[room] == edge['to_id'] and edge['relation_type'] == 'INSIDE':
                        obs_dict['in_' + room] = 1
            
                for item in self.items:
                    if self.env.target_id[item] == edge['to_id']:
                        if edge['relation_type'] == 'CLOSE':
                            obs_dict['close_to_' + item] = 1
                    
                        elif edge['relation_type'] in ['HOLDS_LH', 'HOLDS_RH'] and item in ['chips', 'milk']:
                            obs_dict['hold_' + item] = 1
                        
                        elif edge['relation_type'] == 'FACING' and item == 'tv':
                            obs_dict['is_face_' + item] = 1
                        
                        elif edge['relation_type'] == 'ON' and item == 'sofa':
                            obs_dict['is_sit_' + item] = 1
                        
            elif edge['from_id'] == self.env.target_id['chips']:
                if edge['relation_type'] == 'ON' and edge['to_id'] == self.env.target_id['coffeetable']:
                    obs_dict['chips_on_coffeetable'] = 1

            elif edge['from_id'] == self.env.target_id['milk']:
                if edge['relation_type'] == 'ON' and edge['to_id'] == self.env.target_id['coffeetable']:
                    obs_dict['milk_on_coffeetable'] = 1

        return [value for value in obs_dict.values()], obs_dict
    
    def create_action_list(self):
        self.action_list = [
            '[WALK] <livingroom> ({})'.format(self.env.target_id['livingroom']),
            '[WALK] <kitchen> ({})'.format(self.env.target_id['kitchen']),
            '[WALK] <bathroom> ({})'.format(self.env.target_id['bathroom']),
            '[WALK] <bedroom> ({})'.format(self.env.target_id['bedroom']),
            
            '[WALK] <chips> ({})'.format(self.env.target_id['chips']),
            '[WALK] <milk> ({})'.format(self.env.target_id['milk']),
            '[WALK] <coffeetable> ({})'.format(self.env.target_id['coffeetable']),
            '[WALK] <television> ({})'.format(self.env.target_id['tv']),
            '[WALK] <sofa> ({})'.format(self.env.target_id['sofa']),

            '[Grab] <chips> ({})'.format(self.env.target_id['chips']),
            '[Grab] <milk> ({})'.format(self.env.target_id['milk']),
            
            '[putback] <chips> ({}) <coffeetable> ({})'.format(self.env.target_id['chips'], self.env.target_id['coffeetable']),
            '[putback] <milk> ({}) <coffeetable> ({})'.format(self.env.target_id['milk'], self.env.target_id['coffeetable']),

            '[SWITCHON] <television> ({})'.format(self.env.target_id['tv']),
            '[SWITCHOFF] <television> ({})'.format(self.env.target_id['tv']),

            '[SIT] <sofa> ({})'.format(self.env.target_id['sofa']),
            '[StandUp]',
        ]

        self.action_template = [
            "walk to the living room", # 0 Walk-LivingRoom
            "walk to the kitchen", # 1 Walk-Kitchen
            "walk to the bathroom", # 2 Walk-Bathroom
            "walk to the bedroom", # 3 Walk-Bedroom

            "walk to the chips", # 4 Walk-Chips
            "walk to the milk", # 5 Walk-Milk
            'walk to the coffee table', # 6 Walk-CoffeeTable
            'walk to the TV', # 7 Walk-TV
            'walk to the sofa', # 8 Walk-Sofa

            "grab the chips", # 9
            "grab the milk", # 10

            'put the chips on the coffee table', # 11
            'put the milk on the coffee table', # 12

            "turn on the TV", # 13
            "turn off the TV", # 14

            "sit on the sofa", # 15
            "stand up from the sofa" # 16
        ]

    def obs2text(self, obs):
        """
        {
            'in_kitchen': 1,
            'in_bathroom': 0,
            'in_bedroom': 0,
            'in_livingroom': 0,
            'see_chips': 1,
            'close_to_chips': 0,
            'hold_chips': 0,
            'chips_on_coffeetable': 0,
            'see_milk': 1,
            'close_to_milk': 0,
            'hold_milk': 0,
            'milk_on_coffeetable': 0,
            'see_tv': 0,
            'close_to_tv': 0,
            'is_face_tv': 0,
            'is_tv_on': 0,
            'see_sofa': 0,
            'close_to_sofa': 0,
            'is_sit_sofa': 0,
            'see_coffeetable': 0,
            'close_to_coffeetable': 0
        }
        """

        text = ""

        if self.debug:
            in_kitchen = int(obs['in_kitchen'])
            in_bathroom = int(obs['in_bathroom'])
            in_bedroom = int(obs['in_bedroom'])
            in_livingroom = int(obs['in_livingroom'])

            see_chips = int(obs['see_chips'])
            close_to_chips = int(obs['close_to_chips'])
            hold_chips = int(obs['hold_chips'])
            chips_on_coffeetable = int(obs['chips_on_coffeetable'])

            see_milk = int(obs['see_milk'])
            close_to_milk = int(obs['close_to_milk'])
            hold_milk = int(obs['hold_milk'])
            milk_on_coffeetable = int(obs['milk_on_coffeetable'])

            see_tv = int(obs['see_tv'])
            close_to_tv = int(obs['close_to_tv'])
            is_face_tv = int(obs['is_face_tv'])
            is_tv_on = int(obs['is_tv_on'])

            see_sofa = int(obs['see_sofa'])
            close_to_sofa = int(obs['close_to_sofa'])
            is_sit_sofa = int(obs['is_sit_sofa'])

            see_coffeetable = int(obs['see_coffeetable'])
            close_to_coffeetable = int(obs['close_to_coffeetable'])
        else:
            in_kitchen = obs[0]
            in_bathroom = obs[1]
            in_bedroom = obs[2]
            in_livingroom = obs[3]

            see_chips = obs[4]
            close_to_chips = obs[5]
            hold_chips = obs[6]
            chips_on_coffeetable = obs[7]

            see_milk = obs[8]
            close_to_milk = obs[9]
            hold_milk = obs[10]
            milk_on_coffeetable = obs[11]

            see_tv = obs[12]
            close_to_tv = obs[13]
            is_face_tv = obs[14]
            is_tv_on = obs[15]

            see_sofa = obs[16]
            close_to_sofa = obs[17]
            is_sit_sofa = obs[18]

            see_coffeetable = obs[19]
            close_to_coffeetable = obs[20]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {} "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        ########################################template2####################################
        # template for kitchen
        object_text = ""

        action_list = []

        if in_kitchen:

            if see_chips and see_milk:
                object_text += "and notice chips and milk. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                elif hold_chips and not hold_milk:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not close to you. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips are not close to you. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are close to you. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            5,
                        ]

                    object_text += "Currently, you are not grabbing anything in hand. "

            elif see_chips and not see_milk:
                object_text += "and only notice chips. "

                if hold_chips:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                        ]

            elif not see_chips and see_milk:
                object_text += "and notice milk. "

                if hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                        ]

            else:
                object_text += "and notice nothing. "

                action_list = [
                    0,
                    2,
                    3,
                ]

        elif in_livingroom:

            object_text += "and you notice a coffee table, a TV and a sofa. "

            assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "You are next to more than one object from coffee table, TV and sofa."
            assert see_coffeetable + see_tv + see_sofa >= 3, "You don't see coffee table, TV and sofa."

            if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                object_text += "They are not close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "
                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "
                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                action_list = [
                    1,
                    2,
                    3,
                    6,
                    7,
                    8
                ]

            if close_to_coffeetable:

                if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                    object_text += "The TV is not close to you. "
                else:
                    object_text += "The coffee table is close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        7,
                        8,
                        11,
                        12
                    ]
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += "Currently, you have grabbed the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            12
                        ]

                    else:
                        object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    if not milk_on_coffeetable:
                        object_text += "Currently, you have grabbed the chips in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            11
                        ]

                    else:
                        object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                    ]

            if close_to_tv:
                if is_tv_on:
                    object_text += "The sofa is not close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                    ]

                else:
                    object_text += "The TV is close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            8,
                        ]

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            8,
                            13,
                            14
                        ]
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            8,
                            13,
                            14
                        ]

            if close_to_sofa:

                if not is_sit_sofa:
                    object_text += "The sofa is close to you. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                            15,
                            16
                        ]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                        ]

                else:
                    object_text += "You are sitting on the sofa. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]

        elif in_bedroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_bathroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]

        text += object_text

        # template for target
        # target_template = "If you want to have chips and milk to enjoy while you watch TV. "
        # target_template = "You want to get some chips and milk and put the chips on the coffee table, then sit on the sofa and enjoy them while watching TV."
        target_template = "In order to enjoy the chips and the milk while watching TV, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        action_template = [self.action_template[i] for i in action_list]

        return text, action_template
