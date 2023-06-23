from gym import Wrapper
import random
class Obs2TextWrapper(Wrapper):
    def __init__(self, env):
        super(Obs2TextWrapper, self).__init__(env)

    def reset(self): 
        return self.obs2text(self.env.reset())
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.obs2text(obs), reward, done, info

    def obs2text(self):
        prompt = ["good morning", "one plus one", "How are you"]
        return random.choice(prompt)
