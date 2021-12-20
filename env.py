from typing import Optional

import gym
import numpy as np
from gym import spaces

from PIL_helper import draw_circle, draw_mat


class World(gym.Env):
    def __init__(self, file_path):
        self.world_map = np.load(file_path)
        self.height, self.width = self.world_map.shape
        # self.world_field = np.zeros(shape=(self.height, self.width))
        self.start = (100, 50)
        self.end = (400, 550)
        self.agents = [self.start]

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
        pil_image = draw_mat(self.world_map)
        draw_circle(*self.start, 10, img=pil_image, color='red')
        draw_circle(*self.end, 10, img=pil_image, color='blue')
        pil_image.show()

    # close viewer if exists
    def close(self):
        pass


# main
w = World('maps/map1.npy')
print(w.height,w.width)
w.render()
