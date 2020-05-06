import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SmartEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('Environment initialized!')
        pass

    def step(self, action):
        print('Step successful!')
        pass

    def reset(self):
        print('Environment reset!')
        pass

    def render(self, mode='human', close=False):
        pass
