from typing import Optional

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt, cm

from PIL_helper import draw_circle, draw_mat


class World(gym.Env):
    def __init__(self, file_path):
        self.world_map = np.load(file_path)
        self.height, self.width = self.world_map.shape
        # self.world_field = np.zeros(shape=(self.height, self.width))
        self.start = (50, 50)
        self.end = (300, 300)
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

    def potential_field(self):
        K_attr = 0.005
        x_range, y_range = np.arange(self.width), np.arange(self.height)
        X, Y = np.meshgrid(x_range, y_range)
        x_end, y_end = self.end
        DIST_SQUARE = (X - x_end) ** 2 + (Y - y_end) ** 2
        U_attract = 0.5 * K_attr * DIST_SQUARE
        self.render_potential_field(X, Y, U_attract)

    def render_potential_field(self, X, Y, Z):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        # ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        #  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


# main
w = World('maps/map1.npy')
w.potential_field()
