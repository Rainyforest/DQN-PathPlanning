from typing import Any

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt, cm
from pydantic import BaseModel
from scipy import ndimage

from PIL_helper import draw_point, draw_mat


class State(BaseModel):
    agent: tuple
    observation: Any


class Reward(BaseModel):
    collision_reward: float
    goal_reward: float
    time_reward: float
    potential_reward: float


class Config(BaseModel):
    OBSERVATION_RADIUS: int = 100
    K_attractive = 0.005
    K_repulsive = 500
    K_blackhole = 0.0015
    REPULSIVE_THRESHOLD = 200
    BLACKHOLE_THRESHOLD = 600 * 2 ** 0.5
    OBSTACLE_POTENTIAL = 100


class World(gym.Env):
    def __init__(self, file_path, config=Config()):
        self.world_map = np.load(file_path)
        self.height, self.width = self.world_map.shape
        self.start = (50, 50)
        self.end = (300, 300)
        self.potential_field = self.potential_field()
        self.config = config

        # RL Traininig
        self.seed = 0
        self.action_space = spaces.Discrete(9)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # list [pos of agents, their observable potential field]
        self.state = State(agent=self.start, observation=self.get_new_observation())

    # input action return state
    def step(self, action: int):
        assert self.action_space.contains(action)
        #  7 0 1
        #  6 8 2 (Move Directions)
        #  5 4 3
        # calculate state
        self.state.agent = self.get_new_agent(action)
        self.state.observation = self.get_new_observation()
        print(self.state)
        # calculate reward

    def get_new_agent(self, action):
        action_vector_map = {
            0: (0, -1),
            1: (1, -1),
            2: (1, 0),
            3: (1, 1),
            4: (0, 1),
            5: (-1, 1),
            6: (-1, 0),
            7: (-1, -1),
            8: (0, 0)
        }
        del_x, del_y = action_vector_map.get(action, (0, 0))
        x, y = self.state.agent
        return x + del_x, y + del_y

    # return to origin state
    def reset(self):
        np.random.seed(self.seed)
        self.state.agent = np.random.randint(0, high=self.width), np.random.randint(0, high=self.height)
        self.state.observation = self.get_new_observation()

    # display some 2D rendering
    def render(self, mode="human"):
        self.render_path_map()
        U = self.potential_field
        self.render_potential_field(U)
        plt.show()

    # close viewer if exists
    def close(self):
        pass

    def get_new_observation(self):
        pf = self.potential_field
        return None

    def potential_field(self):
        K_attractive = self.config.K_attractive
        K_repulsive = self.config.K_repulsive
        K_blackhole = self.config.K_blackhole
        REPULSIVE_THRESHOLD = self.config.REPULSIVE_THRESHOLD
        BLACKHOLE_THRESHOLD = self.config.BLACKHOLE_THRESHOLD
        OBSTACLE_POTENTIAL = self.config.OBSTACLE_POTENTIAL

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
        D_OBS = np.where(D_OBS == 0,
                         float('inf'),
                         D_OBS)
        U_repulsive = np.clip(D_OBS, 0, OBSTACLE_POTENTIAL)

        U_blackhole = np.where(D <= BLACKHOLE_THRESHOLD,
                               -0.5 * K_blackhole * ((BLACKHOLE_THRESHOLD - D) ** 2),
                               0)

        U = U_attract + U_repulsive + U_blackhole
        return U

    def render_path_map(self):
        plt.title("Path Map")
        draw_mat(self.world_map)
        draw_point(*self.start, 10, color='crimson', label='start point')
        draw_point(*self.end, 10, color='aquamarine', label='end point')
        plt.legend()

    def render_potential_field(self, Z):
        xrange, yrange = np.arange(self.width), np.arange(self.height)
        X, Y = np.meshgrid(xrange, yrange)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, rcount=self.height / 10, ccount=self.width / 10)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)


# main
w = World('maps/map1.npy')
# w.render()
# w.potential_field()
w.step(0)
