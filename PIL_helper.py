from PIL.Image import fromarray
from PIL.ImageDraw import Draw


def draw_mat(mat):
    pixel_map = (1 - mat) * 255
    pil_image = fromarray(pixel_map)
    pil_image = pil_image.convert('RGB')
    return pil_image


def draw_circle(X, Y, r, img, color='black'):
    draw = Draw(img)
    draw.ellipse([(X - r, Y - r), (X + r, Y + r)], fill=color, outline=color)
