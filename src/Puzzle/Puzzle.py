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

from Puzzle.tuple_helper import equals_tuple, add_tuple, sub_tuple, is_neigbhor, corner_puzzle_alignement, display_dim


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

        while len(border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:
                for c_edge in c_piece.edges_:
                    if not c_edge.connected:
                        self.find_matching_piece(c_edge, border_pieces)
                        is_valid = False
                        break
                if not is_valid:
                    break
            break

    def find_matching_piece(self, ref_edge, candidate_pieces):
        r_x1, r_y1 = ref_edge.shape[0]
        r_x2, r_y2 = ref_edge.shape[-1]
        r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

        if ref_edge.type == TypeEdge.HOLE:
            for candidate_piece in candidate_pieces:
                for edge in candidate_piece.edges_:
                    if edge.type == TypeEdge.HEAD:
                        x1, y1 = edge.shape[-1]
                        x2, y2 = edge.shape[0]
                        theta = math.atan2(y2-y1, x2-x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape)@rot_matrix

                        translate = np.array([r_x1-rotated_pixels[-1][0], r_y1-rotated_pixels[-1][1]])
                        translated_pixels = rotated_pixels + translate

                        import matplotlib.pyplot as plt
                        #plt.scatter(rotated_pixels[...,0],rotated_pixels[...,1], color='r')
                        plt.scatter(translated_pixels[...,0], translated_pixels[..., 1], color='g')
                        plt.scatter(ref_edge.shape[...,0], ref_edge.shape[..., 1], color='k')
                        plt.show()

                        x1, y1 = edge.shape[0]
                        x2, y2 = edge.shape[-1]
                        theta = math.atan2(y2 - y1, x2 - x1)

                        theta_diff = r_theta - theta

                        rot_matrix = np.array([[math.cos(theta_diff), math.sin(theta_diff)],
                                               [-math.sin(theta_diff), math.cos(theta_diff)]])

                        rotated_pixels = np.array(edge.shape) @ rot_matrix

                        translate = np.array([r_x1 - rotated_pixels[0][0], r_y1 - rotated_pixels[0][1]])
                        translated_pixels = rotated_pixels + translate

                        import matplotlib.pyplot as plt
                        # plt.scatter(rotated_pixels[...,0],rotated_pixels[...,1], color='r')
                        plt.scatter(translated_pixels[..., 0], translated_pixels[..., 1], color='g')
                        plt.scatter(ref_edge.shape[..., 0], ref_edge.shape[..., 1], color='k')
                        plt.show()
                        # import sys
                        # sys.exit()






        if ref_edge.type == TypeEdge.HEAD:
            self.log("HOLE")