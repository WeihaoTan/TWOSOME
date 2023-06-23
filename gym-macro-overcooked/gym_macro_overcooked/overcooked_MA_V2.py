import numpy as np
from queue import PriorityQueue
from gym import spaces
from .items import Tomato, Onion, Lettuce, Plate, Knife, Delivery, Agent, Food
from .overcooked_V3 import Overcooked_V3
from .mac_agent import MacAgent
import random

# simplify action space for ppo baseline

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
ACTIONIDX = {"right": 0, "down": 1, "left": 2, "up": 3, "stay": 4}
PRIMITIVEACTION =["right", "down", "left", "up", "stay"]

class AStarAgent(object):
    def __init__(self, x, y, g, dis, action, history_action, pass_agent):

        """
        Parameters
        ----------
        x : int
            X position of the agent.
        y : int
            Y position of the agent.
        g : int 
            Cost of the path from the start node to n.
        dis : int
            Distance of the current path.
            g + h
        pass_agent : int
            Whether there is other agent in the path.
        """

        self.x = x
        self.y = y
        self.g = g
        self.dis = dis
        self.action = action
        self.history_action = history_action
        self.pass_agent = pass_agent

    def __lt__(self, other):
        if self.dis != other.dis:
            return self.dis <= other.dis
        else:
            return self.pass_agent <= other.pass_agent

class Overcooked_MA_V2(Overcooked_V3):

    """
    Overcooked Domain Description
    ------------------------------
    ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
    map_type = ["A", "B", "C"]

    Only macro-action is available in this env.
    Macro-actions in map A:
    ["get tomato", "get lettuce", "get onion", "get plate 1", "go to knife 1", "go to knife 2", "deliver", "chop"]
    Macro-actions in map B/C:
    ["get tomato", "get lettuce", "get onion", "get plate 1", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter"]
    
    1) Agent is allowed to pick up/put down food/plate on the counter;
    2) Agent is allowed to chop food into pieces if the food is on the cutting board counter;
    3) Agent is allowed to deliver food to the delivery counter;
    4) Only unchopped food is allowed to be chopped;
    """
        
    def __init__(self, grid_dim, task, rewardList, map_type = "A", n_agent = 2, obs_radius = 2, mode = "vector", debug = False):

        """
        Parameters
        ----------
        gird_dim : tuple(int, int)
            The size of the grid world([7, 7]/[9, 9]).
        task : int
            The index of the target recipe.
        rewardList : dictionary
            The list of the reward.
            e.g rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
        map_type : str 
            The type of the map(A/B/C).
        n_agent: int
            The number of the agents.
        obs_radius: int
            The radius of the agents.
        mode: string
            The type of the observation(vector/image).
        debug : bool
            Whehter print the debug information.
        """

        super().__init__(grid_dim, task, rewardList, map_type, n_agent, obs_radius, mode, debug)
        self.macroAgent = []
        self._createMacroAgents()
        self.macroActionItemList = []
        self._createMacroActionItemList()

        if map_type == "A":
            #self.macroActionName = ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop"]
            self.macroActionName = ["get tomato", "get lettuce", "get onion", "get plate 1", "go to knife 1", "go to knife 2", "deliver", "chop"]
        else:
            #self.macroActionName = ["stay", "get tomato", "get lettuce", "get onion", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter"]
            self.macroActionName = ["get tomato", "get lettuce", "get onion", "get plate 1", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter"]
        self.action_space = spaces.Discrete(len(self.macroActionName))

        if self.xlen == 7 and self.ylen == 7:
            if self.mapType == "B":
                self.counterSequence = [3, 2, 4, 1, 5]
            elif self.mapType == "C":
                self.counterSequence = [3, 2, 4, 1]
        elif self.xlen == 9 and self.ylen == 9:
            if self.mapType == "B":
                self.counterSequence = [4, 3, 5, 2, 6, 1, 7]
            elif self.mapType == "C":
                self.counterSequence = [4, 3, 5, 2, 6, 1]

    def _createMacroAgents(self):
        for agent in self.agent:
            self.macroAgent.append(MacAgent())

    def _createMacroActionItemList(self):
        self.macroActionItemList = []
        for key in self.itemDic:
            if key != "agent":
                self.macroActionItemList += self.itemDic[key]

    def macro_action_sample(self):
        macro_actions = []
        for agent in self.agent:
            macro_actions.append(random.randint(0, self.action_space.n - 1))
        return macro_actions     

    def build_agents(self):
        raise

    def build_macro_actions(self):
        raise

    def _findPOitem(self, agent, macro_action):
    
        """
        Parameters
        ----------
        agent : Agent
        macro_action: int

        Returns
        -------
        x : int
            X position of the item in the observation of the agent.
        y : int
            Y position of the item in the observation of the agent.
        """
        #{"tomato": self.tomato, "lettuce": self.lettuce, "onion": self.onion, "plate": self.plate, "knife": self.knife, "delivery": self.delivery, "agent": self.agent}

        foodIdx = self.macroActionName.index("get plate 1")
        if macro_action < foodIdx:
            idx = macro_action * 3
        else:
            idx = macro_action * 2 + foodIdx
        return int(agent.obs[idx] * self.xlen), int(agent.obs[idx + 1] * self.ylen)

    def reset(self):
                
        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """

        super().reset()
        for agent in self.macroAgent:
            agent.reset()
        return self._get_macro_obs()

    def run(self, macro_actions):

        """
        Parameters
        ----------
        macro_actions: list
            macro_action for each agent

        Returns
        -------
        macro_obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """
        mac_done = False
        reward = 0
        macro_action_steps = 0
        discount_factor = 1

        while not mac_done:
            macro_action_steps += 1
            if not isinstance(macro_actions, list):
                macro_actions = [macro_actions]
            actions = self._computeLowLevelActions(macro_actions)
            
            obs, rewards, terminate, info = self.step(actions)
            reward += discount_factor * rewards[0] if self.n_agent == 1 else rewards

            self._checkMacroActionDone()
            self._checkCollision(info)
            cur_mac = self._collectCurMacroActions()
            mac_done = self._computeMacroActionDone()

            self._createMacroActionItemList()

            mac_done = mac_done[0]
            discount_factor *= 0.99
        info['macro_action_steps'] = macro_action_steps
        return  self._get_macro_obs(), reward, terminate, info

    def _checkCollision(self, info):
        for idx in info["collision"]:
            self.macroAgent[idx].cur_macro_action_done = True

    def _checkMacroActionDone(self):
        # loop each agent
        for idx, agent in enumerate(self.agent):
            if not self.macroAgent[idx].cur_macro_action_done:
                macro_action = self.macroAgent[idx].cur_macro_action
                if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]:
                    target_x, target_y = self._findPOitem(agent, macro_action)

                    macroAction2ItemName = {"get tomato": "tomato", "get lettuce": "lettuce", "get onion": "onion"}
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        for knife in self.knife:
                            if knife.x == target_x and knife.y == target_y:
                                food = self._findItem(target_x, target_y, macroAction2ItemName[self.macroActionName[macro_action]])
                                if not food.chopped:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    break
                elif self.macroActionName[macro_action] == "deliver" and not agent.holding:
                    target_x, target_y = self._findPOitem(agent, macro_action)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                elif self.mapType in ["B", "C"] and self.macroActionName[macro_action] == "go to counter " and not agent.holding:
                    target_x = 0
                    target_y = int(self.ylen // 2)
                    findEmptyCounter = False
                    for i in self.counterSequence:
                        if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                            target_x = i
                            findEmptyCounter = True
                            break
                    if findEmptyCounter:
                        if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                            self.macroAgent[idx].cur_macro_action_done = True
                    else:
                        self.macroAgent[idx].cur_macro_action_done = True

                if self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]\
                    or self.macroActionName[macro_action] in ["get plate 1"]:
                        target_x, target_y = self._findPOitem(agent, macro_action)
                        macroAction2Item = {"get tomato": self.tomato[0], "get lettuce": self.lettuce[0], "get onion": self.onion[0], "get plate 1": self.plate[0]}
                        item = macroAction2Item[self.macroActionName[macro_action]]
                        if target_x != item.x or target_y != item.y:
                            self.macroAgent[idx].cur_macro_action_done = True

    def _computeLowLevelActions(self, macro_actions):

        """
        Parameters
        ----------
        macro_actions : int | List[..]
            The discrete macro-actions index for the agents. 

        Returns
        -------
        primitive_actions : int | List[..]
            The discrete primitive-actions index for the agents. 
        """

        primitive_actions = []
        # loop each agent
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                self.macroAgent[idx].cur_macro_action = macro_actions[idx]
                macro_action = macro_actions[idx]
                self.macroAgent[idx].cur_macro_action_done = False
            else:
                macro_action = self.macroAgent[idx].cur_macro_action

            primitive_action = ACTIONIDX["stay"]

            # if macro_action == 0:
            #     self.macroAgent[idx].cur_macro_action_done = True
            # elif self.macroActionName[macro_action] == "chop":
            if self.macroActionName[macro_action] == "chop":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]] 
                    if new_name == "knife":
                        knife = self._findItem(new_x, new_y, new_name)
                        if isinstance(knife.holding, Food):
                            if not knife.holding.chopped:
                                primitive_action = action
                                self.macroAgent[idx].cur_chop_times += 1
                                if self.macroAgent[idx].cur_chop_times >= 3:
                                    self.macroAgent[idx].cur_macro_action_done = True
                                    self.macroAgent[idx].cur_chop_times = 0
                                break
                if primitive_action == ACTIONIDX["stay"]:
                    self.macroAgent[idx].cur_macro_action_done = True
            elif self.macroActionName[macro_action] == "deliver" and agent.x == 1 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                primitive_action = ACTIONIDX["right"]
            elif self.mapType in ["B", "C"] and self.macroActionName[macro_action] == "go to counter":
                findEmptyCounter = False
                target_x = 0
                target_y = int(self.ylen // 2)
                for i in self.counterSequence:
                    if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                        target_x = i
                        findEmptyCounter = True
                        break
                if findEmptyCounter:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                else:
                    primitive_action = ACTIONIDX["stay"]
                    self.macroAgent[idx].cur_macro_action_done = True
            else:
                target_x, target_y = self._findPOitem(agent, macro_action)

                inPlate = False
                if self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]:
                    if (target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) \
                        or self.obs_radius == 0:
                        for plate in self.plate:
                            if plate.x == target_x and plate.y == target_y:
                                primitive_action = ACTIONIDX["stay"]
                                self.macroAgent[idx].cur_macro_action_done = True
                                inPlate = True
                                break
                if inPlate:
                    primitive_actions.append(primitive_action)
                    continue
            
                if target_x == 1 and target_y == 0 and agent.x == 3 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                    primitive_action = ACTIONIDX["right"]
                elif ITEMNAME[agent.pomap[target_x][target_y]] == "agent" \
                    and ((target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) or self.obs_radius == 0):
                    self.macroAgent[idx].cur_macro_action_done = True
                else:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if primitive_action == ACTIONIDX["stay"]:
                        self.macroAgent[idx].cur_macro_action_done = True
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        self.macroAgent[idx].cur_macro_action_done = True
                        if self.macroActionName[macro_action] in ["get plate 1"] and agent.holding:
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    self.macroAgent[idx].cur_macro_action_done = False
                                else:
                                    primitive_action = ACTIONIDX["stay"]
                        
                        if self.macroActionName[macro_action] in ["go to knife 1", "go to knife 2"] and not agent.holding:
                            primitive_action = ACTIONIDX["stay"]

                        if self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]:
                                for knife in self.knife:
                                    if knife.x == target_x and knife.y == target_y:
                                        if isinstance(knife.holding, Food):
                                            if not knife.holding.chopped:
                                                primitive_action = ACTIONIDX["stay"]
                                                break
                        
                        if self.macroActionName[macro_action] in ["get tomato", "get lettuce", "get onion"]\
                            or self.macroActionName[macro_action] in ["get plate 1"]:
                            macroAction2Item = {"get tomato": self.tomato[0], "get lettuce": self.lettuce[0], "get onion": self.onion[0], "get plate 1": self.plate[0]}
                            item = macroAction2Item[self.macroActionName[macro_action]]
                            if target_x != item.x or target_y != item.y:
                                 primitive_action = ACTIONIDX["stay"]

            primitive_actions.append(primitive_action)
        return primitive_actions
           
    # A star
    def _navigate(self, agent, target_x, target_y):

        """
        Parameters
        ----------
        agent : Agent
            The current agent.
        target_x : int
            X position of the target item.
        target_y : int
            Y position of the target item.                 

        Returns
        -------
        action : int
            The primitive-action for the agent to choose.
        """

        direction = [(0,1), (0,-1), (1,0), (-1,0)]
        actionIdx = [0, 2, 1, 3]

        # make the agent explore up and down first to aviod deadlock when going to the knife
        q = PriorityQueue()
        q.put(AStarAgent(agent.x, agent.y, 0, self._calDistance(agent.x, agent.y, target_x, target_y), None, [], 0))
        isVisited = [[False for col in range(self.ylen)] for row in range(self.xlen)]
        isVisited[agent.x][agent.y] = True

        while not q.empty():
            aStarAgent = q.get()

            for action in range(4):
                new_x = aStarAgent.x + direction[action][0]
                new_y = aStarAgent.y + direction[action][1]
                new_name = ITEMNAME[agent.pomap[new_x][new_y]] 

                if not isVisited[new_x][new_y]:
                    init_action = None
                    if aStarAgent.action is not None:
                        init_action = aStarAgent.action
                    else:
                        init_action = actionIdx[action]

                    if new_name == "space" or new_name == "agent":
                        pass_agent = 0
                        if new_name == "agent":
                            pass_agent = 1
                        g = aStarAgent.g + 1
                        f = g + self._calDistance(new_x, new_y, target_x, target_y)
                        q.put(AStarAgent(new_x, new_y, g, f, init_action, aStarAgent.history_action + [actionIdx[action]], pass_agent))
                        isVisited[new_x][new_y] = True
                    if new_x == target_x and new_y == target_y:
                        return init_action
        #if no path found, stay
        return ACTIONIDX["stay"]

    def _calDistance(self, x, y, target_x, target_y):
        return abs(target_x - x) + abs(target_y - y)
    
    def _calItemDistance(self, agent, item):
        return abs(item.x - agent.x) + abs(item.y - agent.y)

    def _collectCurMacroActions(self):
        # loop each agent
        cur_mac = []
        for agent in self.macroAgent:
            cur_mac.append(agent.cur_macro_action)
        return cur_mac

    def _computeMacroActionDone(self):
        # loop each agent
        mac_done = []
        for agent in self.macroAgent:
            mac_done.append(agent.cur_macro_action_done)
        return mac_done

    def _get_macro_obs(self):

        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """
        if self.mode == "vector":
            return self._get_macro_vector_obs()
        elif self.mode == "image":
            return self._get_macro_image_obs()
          

    def _get_macro_vector_obs(self):

        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """

        macro_obs = []
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                obs = []
                for item in self.itemList:
                    x = 0
                    y = 0
                    if (item.x >= agent.x - self.obs_radius and item.x <= agent.x + self.obs_radius and item.y >= agent.y - self.obs_radius and item.y <= agent.y + self.obs_radius) \
                        or self.obs_radius == 0:
                        x = item.x / self.xlen
                        y = item.y / self.ylen
                        obs.append(x)
                        obs.append(y)
                        if isinstance(item, Food):
                            obs.append(item.cur_chopped_times / item.required_chopped_times)
                    else:
                        obs.append(0)
                        obs.append(0)
                        if isinstance(item, Food):
                            obs.append(0)
                obs += self.oneHotTask 
                self.macroAgent[idx].cur_macro_obs = obs 
            macro_obs.append(np.array(self.macroAgent[idx].cur_macro_obs))
        if self.n_agent == 1:
            return macro_obs[0]
        return macro_obs

    def _get_macro_image_obs(self):

        """
        Returns
        -------
        macro_image_obs : list
            image observation for each agent.
        """
        
        macro_obs = []
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                frame = self.game.get_image_obs()
                if self.obs_radius > 0:
                    old_image_width, old_image_height, channels = frame.shape

                    new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
                    new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
                    color = (0,0,0)
                    obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

                    x_center = (new_image_width - old_image_width) // 2
                    y_center = (new_image_height - old_image_height) // 2

                    obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame
                    obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)

                    self.macroAgent[idx].cur_macro_obs = obs 
                else:
                    self.macroAgent[idx].cur_macro_obs = frame 
            macro_obs.append(self.macroAgent[idx].cur_macro_obs)
        return macro_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n