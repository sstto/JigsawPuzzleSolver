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
    if len(ref) > len(trans):
        short = trans
        long = ref
    else :
        short = ref
        long = trans

    idx = np.array(range(len(short)))/len(short)*len(long)
    idx = np.around(idx).astype(int)
    min_ = np.sum(np.sqrt(np.sum((long[idx] - short) ** 2, axis=1)))


    return min_

def show_pyplot(piece) :
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

def show(largeBoard, dx, dy, mx, my):
    largeBoard = np.roll(largeBoard, [-dx, -dy], axis=(0, 1))
    largeBoard = largeBoard[0:mx-dx+100, 0:my-dy+100]

    cv2.imwrite("../result/temp.png", largeBoard)
    temp = cv2.imread("../result/temp.png")
    tempS = cv2.resize(temp, (720, 720))
    cv2.imshow('a', tempS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def boundary(x, y, minX, minY, maxX, maxY):
    if x < minX:
        minX = x
    if y < minY:
        minY = y
    if x > maxX:
        maxX = x
    if y > maxY:
        maxY = y
    return minX, minY, maxX, maxY


def findOppositeSide(piece, dir) :
    if dir==Directions.S :
        for edge in piece.edges_ :
            if edge.direction == Directions.N :
                return edge.type

    elif dir == Directions.N :
        for edge in piece.edges_ :
            if edge.direction == Directions.S :
                return edge.type
    elif dir == Directions.E :
        for edge in piece.edges_ :
            if edge.direction == Directions.W :
                return edge.type
    else:  #dir == Directions.W :
        for edge in piece.edges_ :
            if edge.direction == Directions.E :
                return edge.type

def rotateEdgePixels(r_x, r_y, r_theta, edge, task):
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


class Puzzle():
    """
        Class used to store all informations about the puzzle
    """

    def log(self, *args):
        """ Helper to log informations to the GUI """
        print(' '.join(map(str, args)))

    def get_candidate_by_length(self, ref_edge, pieces, candidate_num=10):
        diff = np.zeros((len(pieces), len(pieces[0].edges_)))
        for i, piece in enumerate(pieces):
            for j, edge in enumerate(piece.edges_):
                diff[i, j] = abs(ref_edge.length - edge.length)
        diff = np.sort(diff, axis =1)[:, 0]
        idx = np.argsort(diff).astype(int)[:candidate_num] if candidate_num>len(pieces) else np.argsort(diff).astype(int)[:]
        return np.array(pieces)[idx]

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
        largeBoard = np.ones((3000, 3000, 3)) * 255
        minX = 2999
        minY = 2999
        maxX = 0
        maxY = 0

        # 분류 작업;
        for piece in self.pieces_:
            if piece.nBorders_ == 0:
                non_border_pieces.append(piece)
            else:
                border_pieces.append(piece)

        # 코너 조각 하나를 complete_pieces 에 넣기;
        for b_piece in border_pieces:
            if b_piece.nBorders_ == 2:

                for pixel in b_piece.img_piece_:
                    largeBoard[pixel.pos] = pixel.color
                    minX, minY, maxX, maxY = boundary(pixel.pos[0], pixel.pos[1], minX, minY, maxX, maxY)

                complete_pieces.append(b_piece)
                border_pieces.remove(b_piece)
                break

        while len(border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:

                for c_edge in c_piece.edges_:
                    if c_piece.nBorders_ != 2 and findOppositeSide(c_piece, c_edge.direction) == TypeEdge.BORDER :
                        continue

                    if not c_edge.connected:
                        candidate_pieces = self.get_candidate_by_length(c_edge, border_pieces, 3)
                        friend, i, theta, trans = self.find_matching_piece(c_edge, candidate_pieces)
                        friend.edges_[i].connected = True
                        c_edge.connected = True
                        is_valid = False
                        border_pieces.remove(friend)
                        complete_pieces.append(friend)

                        rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

                        for edge in friend.edges_ :
                            edge.shape = edge.shape@rot_matrix
                            edge.shape = edge.shape + trans

                        for pixel in friend.img_piece_:
                            pixel.pos = rot_matrix@pixel.pos
                            x = int(pixel.pos[0] + trans[1])
                            y = int(pixel.pos[1] + trans[0])
                            minX, minY, maxX, maxY = boundary(x, y, minX, minY, maxX, maxY)
                            largeBoard[x][y] = pixel.color
                        break
                if not is_valid:
                    break
            show(largeBoard, minX, minY, maxX, maxY)

    def find_matching_piece(self, ref_edge, candidate_pieces):
        r_x1, r_y1 = ref_edge.shape[0]
        r_x2, r_y2 = ref_edge.shape[-1]
        r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

        minimum = 9999999999
        minArg = None
        minEdge = None
        rotAngle = None
        transpose = None

        if ref_edge.type == TypeEdge.HOLE:
            for candidate_piece in candidate_pieces:
                for i, edge in enumerate(candidate_piece.edges_) :
                    if edge.type == TypeEdge.HEAD:
                        #===================Rotation 1=======================
                        translated_pixels1, theta_diff1, translate1 = rotateEdgePixels(r_x1, r_y1, r_theta, edge, -1)

                        #=====================Rotation 2======================
                        translated_pixels2, theta_diff2, translate2 = rotateEdgePixels(r_x1, r_y1, r_theta, edge, 0)

                        #======================Euclidean====================
                        diff1 = euclidean(ref_edge.shape, np.flip(translated_pixels1, axis=0))
                        diff2 = euclidean(ref_edge.shape, translated_pixels2)

                        if min(diff1, diff2) < minimum:
                            minimum = min(diff1, diff2)
                            minArg = candidate_piece
                            minEdge = i
                            if diff1 < diff2 :
                                rotAngle = theta_diff1
                                transpose = translate1
                            else :
                                rotAngle = theta_diff2
                                transpose = translate2

        else:  # ref_edge.type == TypeEdge.HEAD:
            for candidate_piece in candidate_pieces:
                for i, edge in enumerate(candidate_piece.edges_):
                    if edge.type == TypeEdge.HOLE:
                        # ===================Rotation 1=======================
                        translated_pixels1, theta_diff1, translate1 = rotateEdgePixels(r_x1, r_y1, r_theta, edge, -1)

                        # =====================Rotation 2======================
                        translated_pixels2, theta_diff2, translate2 = rotateEdgePixels(r_x1, r_y1, r_theta, edge, 0)

                        # ======================Euclidean====================
                        diff1 = euclidean(ref_edge.shape, np.flip(translated_pixels1, axis=0))
                        diff2 = euclidean(ref_edge.shape, translated_pixels2)

                        if min(diff1, diff2) < minimum:
                            minimum = min(diff1, diff2)
                            minArg = candidate_piece
                            minEdge = i
                            if diff1 < diff2 :
                                rotAngle = theta_diff1
                                transpose = translate1
                            else :
                                rotAngle = theta_diff2
                                transpose = translate2

        return minArg, minEdge, rotAngle, transpose





