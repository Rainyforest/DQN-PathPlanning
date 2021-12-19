import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import Draw


def draw_mat(mat):
    pixel_map = (1 - mat) * 255
    pil_image = fromarray(pixel_map)
    return pil_image


def draw_mat_color(mat):
    pixel_map = (1 - mat) * 255
    # extend color channel to RGB
    rgb_image = np.stack([pixel_map]*3, axis=2)
    pil_image = fromarray(rgb_image, mode='RGB')
    return pil_image


def draw_circle(X, Y, r, img, color='black'):
    draw = Draw(img)
    draw.ellipse([(X - r, Y - r), (X + r, Y + r)], fill=color, outline=color)
