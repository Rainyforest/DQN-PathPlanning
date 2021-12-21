from math import sqrt
from typing import Optional

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt, cm
from scipy import ndimage

from PIL_helper import draw_point, draw_mat


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
        self.render_path_map()
        X, Y, Z = self.potential_field()
        self.render_potential_field(X, Y, Z)
        plt.show()

    # close viewer if exists
    def close(self):
        pass

    def potential_field(self):
        K_attractive = 0.005
        K_repulsive = 500
        K_blackhole = 0.0015
        REPULSIVE_THRESHOLD = 200
        BLACKHOLE_THRESHOLD = 600 * 2 ** 0.5
        OBSTACLE_POTENTIAL = 100

        # get distance matrices
        xrange, yrange = np.arange(self.width), np.arange(self.height)
        X, Y = np.meshgrid(xrange, yrange)
        endX, endY = self.end

        D_SQR = (X - endX) ** 2 + (Y - endY) ** 2
        D = np.sqrt(D_SQR)

        D_OBS = ndimage.distance_transform_edt(1 - self.world_map)  # dist between point and nearest obstacle

        # calculate potential fields
        U_attract = 0.5 * K_attractive * D_SQR

        D_OBS = np.where((D_OBS > 0) & (D_OBS <= REPULSIVE_THRESHOLD),
                         0.5 * K_repulsive * ((1 / D_OBS) - (1 / REPULSIVE_THRESHOLD)) ** 2,
                         0)
        D_OBS = np.where(D_OBS > REPULSIVE_THRESHOLD,
                         0,
                         D_OBS)
        U_repulsive = np.where(D_OBS == 0,
                               OBSTACLE_POTENTIAL,
                               D_OBS)

        U_blackhole = np.where(D <= BLACKHOLE_THRESHOLD,
                               -0.5 * K_blackhole * ((BLACKHOLE_THRESHOLD - D) ** 2),
                               0)

        U = U_attract + U_repulsive + U_blackhole
        return X, Y, U

    def render_path_map(self):
        plt.title("Path Map")
        draw_mat(self.world_map)
        draw_point(*self.start, 10, color='crimson', label='start point')
        draw_point(*self.end, 10, color='aquamarine', label='end point')
        plt.legend()

    def render_potential_field(self, X, Y, Z):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, rcount=self.height/10, ccount=self.width/10)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)


# main
w = World('maps/map1.npy')
w.render()
w.potential_field()
