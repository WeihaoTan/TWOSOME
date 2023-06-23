# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    #parser.add_argument("--env-id", type=str, default="CartPole-v1",
        #help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    parser.add_argument('--env-id',                 action='store',        type=str,             default='VirtualHome-v1',    help='Domain name')
    parser.add_argument('--debug',                  action='store',        type=bool,            default=False,                  help='Whehter print the debug information and render')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):

        action_list = [self.obs2text(item)["action"] for item in x]
        action_ids = [[self.template2action[item] for item in env] for env in action_list]

        prompt_nums = len(action_list)

        logits = self.actor(x)

        actions = []
        log_probs = []
        entroy = []

        for i in range(prompt_nums):
            cur_logits = logits[i][action_ids[i]].view(1, -1)

            probs = Categorical(logits=cur_logits)

            if action is None:
                cur_action = probs.sample()[0]
                cur_action = cur_action.view(-1)
                real_action = torch.tensor([action_ids[i][cur_action.item()]], dtype=torch.int32).to(x.device)
            else:
                real_action = action[i].view(-1)
                cur_action = torch.tensor([action_ids[i].index(real_action.item())], dtype=torch.int32).to(x.device)

            actions.append(real_action)
            log_probs.append(probs.log_prob(cur_action))
            entroy.append(probs.entropy())

        action = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        entroy = torch.cat(entroy)

        return action, log_probs, entroy, self.critic(x)
    def obs2text(self, obs):

        text = ""

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
                        object_text += "The milk is within your immediate reach. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not within your immediate reach. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips is within your immediate reach. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips is not within your immediate reach. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are within your immediate reach. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips is within your immediate reach. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is within your immediate reach. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not within your immediate reach. "

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
                        object_text += "The chips is within your immediate reach. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips is not within your immediate reach. "

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
                        object_text += "The milk is within your immediate reach. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not within your immediate reach. "

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

        target_template = "In order to enjoy the chips and the milk while watching TV, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "reach for the chips",  # 4
            "reach for the milk",  # 5
            'move to the coffee table',  # 6
            'move to the TV',  # 7
            'move to the sofa',  # 8

            "grab the chips",  # 9
            "grab the milk",  # 10

            'put the chips on the coffee table',  # 11
            'put the milk on the coffee table',  # 12

            "turn on the TV",  # 13
            "turn off the TV",  # 14

            "take a seat on the sofa",  # 15
            "stand up from the sofa"  # 16
        ]

        self.template2action = {
            k: i for i, k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"workdir/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    env_params = {
                    'seed': args.seed,
                    'debug': args.debug,
                }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            #TODO: fix this item['episode']['r'] is incorrect
            
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()