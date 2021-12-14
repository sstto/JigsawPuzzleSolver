import numpy as np
from Puzzle.Enums import directions, Directions, TypeEdge, TypePiece, rotate_direction
import math

class PuzzlePiece:

    def __init__(self, edges, img_piece):
        self.position = (0, 0)
        self.edges_ = edges
        self.img_piece_ = img_piece  # List of Pixels
        self.nBorders_ = len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))
        self.type = TypePiece(self.nBorders_)


    def edge_in_direction(self, dir):
        """ Return the edge in the `dir` direction """

        for e in self.edges_:
            if e.direction == dir:
                return e

    def is_border_aligned(self, p2):
        """ Find if a border of the piece is aligned with a border of `p2` """

        for e in self.edges_:
            if e.type == TypeEdge.BORDER and p2.edge_in_direction(e.direction).type == TypeEdge.BORDER:
                return True
        return False


    def border_angle(self, p2, task):

        for e in self.edges_:
            if e.type == TypeEdge.BORDER:

                r_x1, r_y1 = e.shape[0]
                r_x2, r_y2 = e.shape[-1]
                r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

                dir_edge = p2.edge_in_direction(e.direction)
                if dir_edge.type == TypeEdge.BORDER:
                    translated_pixels, theta_diff, translate = rotate_edge_pixels(r_x1, r_y1, r_theta, dir_edge, task)
                    return translated_pixels, theta_diff, translate
        return False

def rotate_edge_pixels(r_x, r_y, r_theta, edge, task):
    # task = -1, task = 0
    if task == -1:
        x1, y1 = edge.shape[-1]
        x2, y2 = edge.shape[0]
    else:
        x1, y1 = edge.shape[0]
        x2, y2 = edge.shape[-1]

    theta = math.atan2(y2 - y1, x2 - x1)

    theta_diff = r_theta - theta

    rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                           [-math.sin(theta_diff), math.cos(theta_diff)]])

    rotated_pixels = np.array(edge.shape) @ rot_matrix

    translate = np.array([r_x - rotated_pixels[task][0], r_y - rotated_pixels[task][1]])
    translated_pixels = rotated_pixels + translate

    return translated_pixels, theta_diff, translate
