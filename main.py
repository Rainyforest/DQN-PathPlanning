import numpy as np


class World:
    def __init__(self):
        self.height, self.width = 10, 10
        self.world_field = np.zeros(shape=(self.height, self.width))
        self.start = (1, 1)
        self.end = (8, 8)
        self.agents = []
        self.world_map = self.init_world_map()

    def init_world_map(self):
        world_map = np.empty([self.height, self.width], dtype=str)
        # world_map = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        world_map[self.start] = 'S'
        world_map[self.end] = 'E'
        return world_map


w = World()
