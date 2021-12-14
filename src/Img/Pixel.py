import math
import numpy as np


class Pixel:

    def __init__(self, pos, color):
        self.pos = pos
        self.color = color


def flatten_colors(pixels):

    colors = np.array(pixels)
    return np.median(colors, axis=0)
