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

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LLMAgent(nn.Module):
    def __init__(self, normalization_mode = 'token', load_path = None, load_8bit = False):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = 'decapoda-research/llama-7b-hf'
        self.lora_r  = 8
        self.lora_alpha = 16
        #self.lora_dropout = 0.05
        self.lora_dropout = 0
        self.lora_target_modules  = ["q_proj", "v_proj",]

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

        #self.action_list = action_list
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
            cache_dir=os.path.join(root, 'weights/llama')
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
        # else:
        #     model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights = None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic


    def save(self, epoch, exp_path):
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        # torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, exp_path):
        print("load model")
        lora_weights = exp_path
        # critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        # self.critic = self._init_critic(critic_weights).to(self.device)
    
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
        action_ids = [[self.template2action[item] for item in env] for env in action_list]
        
        prompt_nums = len(prompt)
        action_nums = [len(item) for item in action_list]

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

        actions = []
        log_probs = []
        entroy = []

        for i in range(prompt_nums):
            logits = action_logits[sum(action_nums[:i]):sum(action_nums[:i+1])].reshape(-1, action_nums[i]).float()

            probs = Categorical(logits=logits)

            if action is None:
                cur_action = probs.sample()[0]
                cur_action = cur_action.view(-1)
                real_action = torch.tensor([action_ids[i][cur_action.item()]], dtype=torch.int32).to(self.device)
            else:
                real_action = action[i].view(-1)
                cur_action = torch.tensor([action_ids[i].index(real_action.item())], dtype=torch.int32).to(self.device)

            actions.append(real_action)
            log_probs.append(probs.log_prob(cur_action))
            entroy.append(probs.entropy())

        action = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        entroy = torch.cat(entroy)

        if return_value:
            return action, log_probs, entroy, self.get_value(prompt)
        else:
            return action, log_probs, entroy, None


    def obs2text(self, obs):

        text = ""

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
                    object_text += "The microwave is within your immediate reach. "
                    action_list = [0, 2, 3, 4, 7, 8, 9]
                else:
                    object_text += "The microwave is not within your immediate reach. "
                    action_list = [0, 2, 3, 4, 5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is within your immediate reach. "
                    action_list = [0, 2, 3, 5, 6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is within your immediate reach. "
                    action_list = [0, 2, 3, 4, 8, 9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not within your immediate reach. "
                    action_list = [0, 2, 3, 4, 5]
                else:
                    if is_microwave_open:
                        action_list = [0, 2, 3, 8, 9]
                    else:
                        action_list = [0, 2, 3, 9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0, 2, 3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]
        elif in_bedroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_livingroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [1, 2, 3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "reach for the pancake",  # 4
            "move to the microwave",  # 5

            "grab the pancake",  # 6

            "put the pancake in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}

