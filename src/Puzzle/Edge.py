import numpy as np
from Puzzle.Enums import TypeEdge, Directions


class Edge:

    def __init__(self, shape, color, type=TypeEdge.HOLE, connected=False, direction=Directions.N):
        self.shape = shape
        self.shape_backup = shape
        self.color = color
        self.type = type
        self.connected = connected
        self.direction = direction
        self.length = self.calculate_length()

    def calculate_length(self):
        shape = np.array(self.shape)
        return np.sum(np.sqrt(np.sum(np.diff(shape, axis=0)**2, axis=1)))

    def color_norm(self, other):
        if len(self.color) > len(other.color):
            short = other.color
            long = self.color
        else:
            short = self.color
            long = other.color

        idx = np.array(range(len(short))) / len(short) * len(long)
        idx = np.around(idx).astype(int)
        diff1 = np.abs(np.array(long)[idx] - np.array(short))
        diff1 = np.mean(diff1)

        diff2 = np.abs(np.array(long)[np.flip(idx)]-np.array(short))
        diff2 = np.mean(diff2)
        return min(diff1, diff2)


    def is_compatible(self, e2):
        """ Helper to determine if two edges are compatible """

        return (self.type == TypeEdge.HOLE and e2.type == TypeEdge.HEAD) or (self.type == TypeEdge.HEAD and e2.type == TypeEdge.HOLE) \
               or self.type == TypeEdge.UNDEFINED or e2.type == TypeEdge.UNDEFINED
