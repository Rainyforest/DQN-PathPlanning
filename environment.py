from typing import Optional

import gym
from gym import spaces


class GridEnv(gym.Env):

    # set constants and configs
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # list [pos of agents, their observable potential field]
        self.state = ()

    # input action return state
    def step(self, action):
        pass

    # return to origin state
    def reset(self, seed: Optional[int] = None):
        pass

    # display some 2D rendering
    def render(self, mode="human"):
        pass

        # close viewer if exists

    def close(self):
        pass
