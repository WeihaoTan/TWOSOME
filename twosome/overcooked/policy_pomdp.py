import sys

import fire
import gradio as gr
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np

from critic import Critic
from torch.distributions.categorical import Categorical
import copy 

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LLMAgent(nn.Module):
    def __init__(self, normalization_mode = 'token', load_path = None, load_8bit = False, task = 3):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = 'Neko-Institute-of-Science/LLaMA-7B-HF'
        self.lora_r  = 8
        self.lora_alpha = 16
        self.lora_dropout = 0
        self.lora_target_modules  = ["q_proj", "v_proj",]

        self.task = task

        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.normalization_mode = normalization_mode

        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.llama = self._init_llama()

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

    def _init_llama(self):
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map="auto",
            cache_dir='weights/llama'
        )

        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        return model

    def _init_actor(self, lora_weights = None):
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.llama, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llama,
                lora_weights,
                torch_dtype=torch.float16,
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        if not self.load_8bit:
            model.half()
        else:
            model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights = None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic


    def save(self, epoch, exp_path):
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, exp_path):
        lora_weights = exp_path
        critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        self.critic = self._init_critic(critic_weights).to(self.device)
    
    def get_value(self, x):
        if type(x) != list:
            x = [self.obs2text(o)["prompt"] for o in x]
            
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"]
        
        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, obs, action=None, is_warmup=False, return_value = True):
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs]
        
        prompt_num = len(prompt)
        action_num = len(action_list[0])

        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        attention_mask = inputs["attention_mask"]
        if is_warmup:
            with torch.no_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.actor(input_ids, attention_mask=attention_mask)
        
        action_list = [item for sublist in action_list for item in sublist]
        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)

        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim = -1) - 1 #delete first token
        sequence_length = torch.sum(attention_mask, dim = -1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [gen_logits[i, start-1:end-1] for i, (start, end) in enumerate(action_index)]
        
        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1==2

        action_logits = action_logits.reshape(-1, action_num).float()

        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()

        if return_value:
            return action, probs.log_prob(action), probs.entropy(), self.get_value(prompt)
        else:
            return action, probs.log_prob(action), probs.entropy(), None


    def obs2text(self, obs):
        if self.task == 3:
            obs = obs.tolist()
            action_list = [
                "pick up the tomato", 
                "pick up the lettuce", 
                "pick up the onion", 
                "take the empty bowl",
                "walk to the first cutting board",
                "walk to the second cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0, 0, 0]
            ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
            raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
            chopped = [False, False, False]
            ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
            sentences = ["There are two fixed cutting boards in the room."]

            item = []
            item_index = []
            agent_pos = obs[17:19]
            first_cutting_board_pos = [1, 0]
            second_cutting_board_pos = [2, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos, "in_second_cutting_board": second_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": [], "in_second_cutting_board": []}
            

            for i in range(4):
                pos = obs[3 * i: 3 * i + 2]
                if  pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)
                    
                if i < 3 and obs[3 * i + 2] == 3:
                    chopped[i] = True
                
                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
                        
                        if len(overlay[k]) > 1:
                            action_list[3] = "take the bowl"

            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."
            elif len(item) == 3:
                template = "You notice {}, {} and {} on the different tables."
            elif len(item) == 4:
                template = "You notice {}, {}, {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize()) 

            cutting_board_index = ["first", "second"]
            cutting_board_name = ["in_first_cutting_board", "in_second_cutting_board"]
            for cindex in range(2):
                if len(overlay[cutting_board_name[cindex]]) == 1:
                    id  = overlay[cutting_board_name[cindex]][0]
                    template = "{} is on the {} cutting board."
                    if id == 3:
                        sentences.append(template.format("a bowl", cutting_board_index[cindex]).capitalize()) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], cutting_board_index[cindex]).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id], cutting_board_index[cindex]).capitalize()) 
                        if agent_pos == [cindex + 1, 1]:
                            action_list[-1] = "chop the " + raw_ingredient[id]
                                
                elif len(overlay[cutting_board_name[cindex]]) > 1:
                    in_plate_item = overlay[cutting_board_name[cindex]][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "A bowl containing chopped {} is on the {} cutting board."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "A bowl containing chopped {} and {} is on the {} cutting board."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "A bowl containing chopped {}, {} and {} is on the {} cutting board."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item], cutting_board_index[cindex]).capitalize()) 

            #in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            #in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the {} cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
                "put the lettuce in the bowl",  
                "put the onion in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the {} cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are standing in front of the {} cutting board, carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format(cutting_board_index[cindex], "a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first") 
                        action_list[5] = action_template.format(raw_ingredient[id], "second") 
                    else:
                        if chopped[id]:
                            sentences.append(template.format(cutting_board_index[cindex], "a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format(cutting_board_index[cindex], "an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[4] = action_template.format(raw_ingredient[id], "first") 
                            action_list[5] = action_template.format(raw_ingredient[id], "second") 
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} in hand."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} and {} in hand."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {}, {} and {} in hand."

                    sentences.append(full_plate_template.format(cutting_board_index[cindex], *[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first") 
                    action_list[5] = action_template.format("bowl", "second") 
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first") 
                        action_list[5] = action_template.format(raw_ingredient[id], "second") 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[4] = action_template.format(raw_ingredient[id], "first") 
                            action_list[5] = action_template.format(raw_ingredient[id], "second") 
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."

                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first") 
                    action_list[5] = action_template.format("bowl", "second")
            sentences.append("To serve the dish of a bowl only containing chopped tomato and lettuce, you should first")
        elif self.task == 0:
            obs = obs.tolist()

            action_list = [
                "pick up the tomato", 
                "take the bowl",
                "walk to the cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0]
            ingredient = ["a tomato", "a bowl"]
            raw_ingredient = ["tomato", "bowl"]
            chopped = [False]
            ori_pos = [[0, 5], [6, 5]]
            sentences = ["There is a fixed cutting board in the room."]
            in_plate = [False, False, False]

            item = []
            item_index = []
            plate_pos = obs[3:5]
            agent_pos = obs[9:11]
            first_cutting_board_pos = [1, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": []}
            
            
            for i in range(2):
                pos = obs[3 * i: 3 * i + 2]
                if  pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)
                    
                if i < 1 and obs[3 * i + 2] == 3:
                    chopped[i] = True
                
                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize()) 

            cutting_board_index = ["first"]
            cutting_board_name = ["in_first_cutting_board"]

            cindex = 0
            if len(overlay[cutting_board_name[cindex]]) == 1:
                id  = overlay[cutting_board_name[cindex]][0]
                template = "{} is on the cutting board."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize()) 
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize()) 
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                    if agent_pos == [cindex + 1, 1]:
                        action_list[-1] = "chop the " + raw_ingredient[id]
                        
                            
            elif len(overlay[cutting_board_name[cindex]]) > 1:

                full_plate_template = "a bowl containing a chopped tomato is on the cutting board."
                sentences.append(full_plate_template.capitalize()) 

            #in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            #in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    id  = overlay["in_agent"][0]
                    action_list[3] = "serve the dish"
                    template = "Currently you are standing in front of the cutting board, carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize()) 
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id]) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id] ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[2] = action_template.format(raw_ingredient[id]) 
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the cutting board, carrying a bowl containing chopped {} in hand."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize()) 
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl") 
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[3] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize()) 
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id]) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[2] = action_template.format(raw_ingredient[id]) 
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())   
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl") 

            sentences.append("To serve the dish of a bowl only containing chopped tomato, you should first")

        return {"prompt": " ".join(sentences), "action": action_list}
