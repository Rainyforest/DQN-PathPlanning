import numpy as np

from PIL_helper import draw_circle, draw_mat_color


class World:
    def __init__(self, file_path):
        self.world_map = np.load(file_path)
        self.height, self.width = self.world_map.shape
        # self.world_field = np.zeros(shape=(self.height, self.width))
        self.start = (50, 50)
        self.end = (550, 550)
        # self.agents = []

    def render(self):
        # pil_image = draw_mat(self.world_map)
        pil_image = draw_mat_color(self.world_map)
        draw_circle(*self.start, 10, img=pil_image, color='red')
        draw_circle(*self.end, 10, img=pil_image, color='blue')
        pil_image.show()


# main
w = World('maps/map1.npy')
w.render()
