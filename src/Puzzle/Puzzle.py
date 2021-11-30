from Puzzle.Distance import diff_match_edges, real_edge_compute, generated_edge_compute
from Puzzle.PuzzlePiece import *
from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2
import math
import numpy as np
from Puzzle.Enums import *
import sys
import scipy
import matplotlib.pyplot as plt
from Puzzle.tuple_helper import equals_tuple, add_tuple, sub_tuple, is_neigbhor, corner_puzzle_alignement, display_dim


def euclidean(ref, trans):
    min1=0
    if len(ref) > len(trans):
        short = trans
        long = ref
    else :
        short = ref
        long = trans

    idx = np.array(range(len(short)))/len(short)*len(long)
    idx = np.around(idx).astype(int)
    min1 = np.sum(np.sqrt(np.sum((long[idx] - short) ** 2, axis=1)))


    trans = np.flip(trans, axis=0)
    min2 = 0
    if len(ref) > len(trans):
        short = trans
        long = ref
    else:
        short = ref
        long = trans

    idx = np.array(range(len(short))) / len(short) * len(long)
    idx = np.around(idx).astype(int)
    min2 = np.sum(np.sqrt(np.sum((long[idx] - short) ** 2, axis=1)))

    return min(min1, min2) # overhead TODO (translated pixel 은 1만)

def show(piece) :
    x_coord = []
    y_coord = []
    color = []

    for pixel in piece.img_piece_:
        y, x = pixel.pos
        x_coord.append(x)
        y_coord.append(y)
        r, g, b = pixel.color
        color.append([b, g, r])

    plt.scatter(x_coord, y_coord, c=np.array(color) / 255.0)
    plt.show()

class Puzzle():
    """
        Class used to store all informations about the puzzle
    """

    def log(self, *args):
        """ Helper to log informations to the GUI """
        print(' '.join(map(str, args)))

    def __init__(self, path):
        """ Extract informations of pieces in the img at `path` and start computation of the solution """

        self.pieces_ = None
        factor = 0.40
        while self.pieces_ is None:
            factor += 0.01
            self.extract = Extractor(path, factor)
            self.pieces_ = self.extract.extract()

        # 조각을 세 가지로 나눈다.;
        # 1. border_pieces : 가장자리 조각 배열;
        # 2. non_border_pieces : 가장자리 아닌 조각열 배열;
        # 3. complete_pieces : 위치를 정한 조각열 배열;
        border_pieces = []
        non_border_pieces = []
        complete_pieces = []

        # 분류 작업;
        for piece in self.pieces_:
            if piece.nBorders_ == 0:
                non_border_pieces.append(piece)
            else:
                border_pieces.append(piece)

        # 코너 조각 하나를 complete_pieces 에 넣기;
        for b_piece in border_pieces:
            if b_piece.nBorders_ == 2:
                complete_pieces.append(b_piece)
                border_pieces.remove(b_piece)
                break

        while len(border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:

                for c_edge in c_piece.edges_:
                    if not c_edge.connected:
                        print('ref')
                        show(c_piece)

                        friend, i = self.find_matching_piece(c_edge, border_pieces)
                        print('friend')
                        show(friend)
                        friend.edges_[i].connected = True
                        c_edge.connected = True
                        is_valid = False
                        break
                if not is_valid:
                    break

            border_pieces.remove(friend)
            complete_pieces.append(friend)


    def find_matching_piece(self, ref_edge, candidate_pieces):
        r_x1, r_y1 = ref_edge.shape[0]
        r_x2, r_y2 = ref_edge.shape[-1]
        r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

        minimum = 9999999999
        minArg = None
        minEdge = None

        if ref_edge.type == TypeEdge.HOLE:
            for candidate_piece in candidate_pieces:
                for i, edge in enumerate(candidate_piece.edges_) :
                    if edge.type == TypeEdge.HEAD:
                        #===================Rotation 1=======================
                        x1, y1 = edge.shape[-1]
                        x2, y2 = edge.shape[0]
                        theta = math.atan2(y2-y1, x2-x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape)@rot_matrix

                        translate = np.array([r_x1-rotated_pixels[-1][0], r_y1-rotated_pixels[-1][1]])
                        translated_pixels1 = rotated_pixels + translate

                        #=====================Rotation 2======================
                        x1, y1 = edge.shape[0]
                        x2, y2 = edge.shape[-1]
                        theta = math.atan2(y2 - y1, x2 - x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape) @ rot_matrix

                        translate = np.array([r_x1 - rotated_pixels[0][0], r_y1 - rotated_pixels[0][1]])
                        translated_pixels2 = rotated_pixels + translate

                        #======================Euclidean====================
                        diff1 = euclidean(ref_edge.shape, translated_pixels1)
                        diff2 = euclidean(ref_edge.shape, translated_pixels2)

                        if min(diff1, diff2) < minimum:
                            minimum = min(diff1, diff2)
                            minArg = candidate_piece
                            minEdge = i

        else:  # ref_edge.type == TypeEdge.HEAD:
            for candidate_piece in candidate_pieces:
                for i, edge in enumerate(candidate_piece.edges_):
                    if edge.type == TypeEdge.HOLE:
                        # ===================Rotation 1=======================
                        x1, y1 = edge.shape[-1]
                        x2, y2 = edge.shape[0]
                        theta = math.atan2(y2 - y1, x2 - x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape) @ rot_matrix

                        translate = np.array([r_x1 - rotated_pixels[-1][0], r_y1 - rotated_pixels[-1][1]])
                        translated_pixels1 = rotated_pixels + translate

                        # =====================Rotation 2======================
                        x1, y1 = edge.shape[0]
                        x2, y2 = edge.shape[-1]
                        theta = math.atan2(y2 - y1, x2 - x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape) @ rot_matrix

                        translate = np.array([r_x1 - rotated_pixels[0][0], r_y1 - rotated_pixels[0][1]])
                        translated_pixels2 = rotated_pixels + translate

                        # ======================Euclidean====================
                        diff1 = euclidean(ref_edge.shape, translated_pixels1)
                        diff2 = euclidean(ref_edge.shape, translated_pixels2)

                        if min(diff1, diff2) < minimum:
                            minimum = min(diff1, diff2)
                            minArg = candidate_piece
                            minEdge = i

        return minArg, minEdge





