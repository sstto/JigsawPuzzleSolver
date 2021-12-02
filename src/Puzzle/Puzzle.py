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
    else:
        short = ref
        long = trans

    idx = np.array(range(len(short)))/len(short)*len(long)
    idx = np.around(idx).astype(int)
    min_ = np.mean(np.sqrt(np.sum((long[idx] - short) ** 2, axis=1)))

    return min_


def show(large_board, dx, dy, mx, my, speed):
    large_board = np.roll(large_board, [-dx, -dy], axis=(0, 1))
    large_board = large_board[0:mx - dx + 100, 0:my - dy + 100]

    cv2.imwrite("../result/temp.png", large_board)
    temp = cv2.imread("../result/temp.png")
    tempS = cv2.resize(temp, (720, 720))
    cv2.imshow('a', tempS)
    cv2.waitKey(speed)
    cv2.destroyAllWindows()


def boundary(x, y, min_x, min_y, max_x, max_y):
    if x < min_x:
        min_x = x
    if y < min_y:
        min_y = y
    if x > max_x:
        max_x = x
    if y > max_y:
        max_y = y
    return min_x, min_y, max_x, max_y


def find_opposite_side(piece, dir):
    if dir == Directions.S:
        for edge in piece.edges_:
            if edge.direction == Directions.N:
                return edge.type

    elif dir == Directions.N:
        for edge in piece.edges_:
            if edge.direction == Directions.S:
                return edge.type
    elif dir == Directions.E:
        for edge in piece.edges_:
            if edge.direction == Directions.W:
                return edge.type
    else:
        # dir == Directions.W :
        for edge in piece.edges_:
            if edge.direction == Directions.E:
                return edge.type


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


def rotate_update(friend, n):
    for edge in friend.edges_:
        edge.direction = rotate_direction(edge.direction, n)


def update_dir(ref_edge_dir, friend, i):
    fr_edge_dir = friend.edges_[i].direction

    rot = np.array([[0, -1], [1, 0]])
    if add_tuple(ref_edge_dir.value, fr_edge_dir.value) == (0, 0):
        return
    elif sub_tuple(ref_edge_dir.value, fr_edge_dir.value) == (0, 0):
        rotate_update(friend, 2)
    else:
        rot_ref_edge_dir = tuple(rot@np.array(ref_edge_dir.value))
        if sub_tuple(rot_ref_edge_dir, fr_edge_dir.value) == (0, 0):
            rotate_update(friend, 3)
        else:
            rotate_update(friend, 1)
    return


def check_duplicate(friend, f_pos, grid):
    for dir in directions:
        pos = add_tuple(f_pos, dir.value)
        if pos in grid:
            rot_dir = rotate_direction(dir, 2)
            grid[pos].edge_in_direction(rot_dir).connected = True

            for f_ed in friend.edges_:
                if f_ed.direction.value == dir.value:
                    f_ed.connected = True
                    break


def find_matching_piece(ref_edge, candidate_pieces, euclidean_num=3):
    r_x1, r_y1 = ref_edge.shape[0]
    r_x2, r_y2 = ref_edge.shape[-1]
    r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

    euclidean_differences_info = []
    euclidean_differences_value = []

    if ref_edge.type == TypeEdge.HOLE:

        for candidate_piece in candidate_pieces:
            for i, edge in enumerate(candidate_piece.edges_):
                if edge.type == TypeEdge.HEAD:
                    #===================Rotation 1=======================
                    translated_pixels1, theta_diff1, translate1 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, -1)

                    #=====================Rotation 2======================
                    translated_pixels2, theta_diff2, translate2 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, 0)

                    #======================Euclidean====================
                    diff1 = euclidean(ref_edge.shape, np.flip(translated_pixels1, axis=0))
                    diff2 = euclidean(ref_edge.shape, translated_pixels2)

                    if diff1 < diff2:
                        euclidean_differences_value.append(diff1)
                        euclidean_differences_info.append((candidate_piece, i, theta_diff1, translate1))
                    else:
                        euclidean_differences_value.append(diff2)
                        euclidean_differences_info.append((candidate_piece, i, theta_diff2, translate2))

    # ref_edge.type == TypeEdge.HEAD:
    else:
        for candidate_piece in candidate_pieces:
            for i, edge in enumerate(candidate_piece.edges_):
                if edge.type == TypeEdge.HOLE:
                    # ===================Rotation 1=======================
                    translated_pixels1, theta_diff1, translate1 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, -1)

                    # =====================Rotation 2======================
                    translated_pixels2, theta_diff2, translate2 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, 0)

                    # ======================Euclidean====================
                    diff1 = euclidean(ref_edge.shape, np.flip(translated_pixels1, axis=0))
                    diff2 = euclidean(ref_edge.shape, translated_pixels2)

                    if diff1 < diff2:
                        euclidean_differences_value.append(diff1)
                        euclidean_differences_info.append((candidate_piece, i, theta_diff1, translate1))
                    else:
                        euclidean_differences_value.append(diff2)
                        euclidean_differences_info.append((candidate_piece, i, theta_diff2, translate2))

    ret = None

    topIdx = np.argsort(euclidean_differences_value)[0:euclidean_num]
    min_ = 99999999
    for idx in topIdx:
        edge_idx = euclidean_differences_info[idx][1]
        edge = euclidean_differences_info[idx][0].edges_[edge_idx]
        color_norm = ref_edge.color_norm(edge)
        if color_norm+euclidean_differences_value[idx]/75 < min_:
            min_ = color_norm+euclidean_differences_value[idx]/75
            ret = euclidean_differences_info[idx]

    return ret


def get_candidate_by_length(ref_edge, pieces, candidate_num=10):
    diff = np.zeros((len(pieces), len(pieces[0].edges_)))
    for i, piece in enumerate(pieces):
        for j, edge in enumerate(piece.edges_):
            diff[i, j] = abs(ref_edge.length - edge.length)
    diff = np.sort(diff, axis=1)[:, 0]
    idx = np.argsort(diff).astype(int)[:candidate_num] \
        if candidate_num > len(pieces) else np.argsort(diff).astype(int)[:]
    return np.array(pieces)[idx]


class Puzzle:
    def __init__(self, path):
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

        grid_completed = dict()

        # 코너 조각 하나를 complete_pieces 에 넣기;
        for b_piece in border_pieces:
            grid_completed[b_piece.position] = b_piece
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
                    if c_piece.nBorders_ != 2 and find_opposite_side(c_piece, c_edge.direction) == TypeEdge.BORDER:
                        continue

                    if not c_edge.connected:
                        candidate_pieces = get_candidate_by_length(c_edge, border_pieces, 3)
                        friend, i, theta, trans = find_matching_piece(c_edge, candidate_pieces)
                        friend.edges_[i].connected = True
                        c_edge.connected = True

                        update_dir(c_edge.direction, friend, i)
                        friend.position = add_tuple(c_piece.position, c_edge.direction.value)
                        grid_completed[friend.position] = friend

                        border_pieces.remove(friend)
                        complete_pieces.append(friend)

                        rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
                        for edge in friend.edges_:
                            edge.shape = edge.shape@rot_matrix
                            edge.shape = edge.shape + trans

                        for pixel in friend.img_piece_:
                            pixel.pos = rot_matrix@pixel.pos
                            x = int(pixel.pos[0] + trans[1])
                            y = int(pixel.pos[1] + trans[0])
                            minX, minY, maxX, maxY = boundary(x, y, minX, minY, maxX, maxY)
                            largeBoard[x][y] = pixel.color

                        is_valid = False
                        break
                if not is_valid:
                    break
            show(largeBoard, minX, minY, maxX, maxY, 0)

#=======================================================================================================================
        for piece in complete_pieces:
            for edge in piece.edges_:
                if not edge.connected:
                    if piece.nBorders_ == 2:
                        edge.connected = True
                    else:
                        if piece.edge_in_direction(rotate_direction(edge.direction, 2)).type != TypeEdge.BORDER:
                            edge.connected = True

        while len(non_border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:
                for c_edge in c_piece.edges_:
                    if not c_edge.connected:
                        candidate_pieces = get_candidate_by_length(c_edge, non_border_pieces, 5)
                        #NEIGHBOR CHANGE==========
                        neighbors = dict()
                        target_position = add_tuple(c_piece.position, c_edge.direction.value)

                        for dir in directions:
                            pos = add_tuple(target_position, dir.value)
                            if pos in grid_completed:
                                op_dir = rotate_direction(dir, 2)
                                tmp = grid_completed[pos].edge_in_direction(op_dir)
                                neighbors[dir] = tmp
                            else:
                                neighbors[dir] = None
                        #NEIGHBOR CHANGE==========
                        neighbor = list(neighbors.values())

                        friend, i, theta, trans = find_matching_piece_center(neighbor, c_edge, candidate_pieces)
                        friend.edges_[i].connected = True
                        c_edge.connected = True
                        is_valid = False

                        update_dir(c_edge.direction, friend, i)
                        friend.position = add_tuple(c_piece.position, c_edge.direction.value)
                        grid_completed[friend.position] = friend

                        non_border_pieces.remove(friend)
                        complete_pieces.append(friend)

                        rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

                        for edge in friend.edges_:
                            edge.shape = edge.shape@rot_matrix
                            edge.shape = edge.shape + trans

                        for pixel in friend.img_piece_:
                            pixel.pos = rot_matrix@pixel.pos
                            x = int(pixel.pos[0] + trans[1])
                            y = int(pixel.pos[1] + trans[0])
                            minX, minY, maxX, maxY = boundary(x, y, minX, minY, maxX, maxY)
                            largeBoard[x][y] = pixel.color

                        check_duplicate(friend, friend.position, grid_completed)
                        break
                if not is_valid:
                    break
            '''
            for k, v in grid_completed.items() :
                print('grid: ', k, )
                for edge in v.edges_ :
                    print('dir: ', edge.direction, 'connected?:' , edge.connected)

                print('\n')
            '''

            show(largeBoard, minX, minY, maxX, maxY, 0)

#=======================================================================================================================


#=======================================================================================================================


def find_matching_piece_center(neighbor, c_edge, candidate_pieces, euclidean_num=3):

    if c_edge.direction == Directions.N:
        finalI = 2
    elif c_edge.direction == Directions.E:
        finalI = 3
    elif c_edge.direction == Directions.S:
        finalI = 0
    else:
        finalI = 1

    nEdge1 = neighbor[0]
    nEdge2 = neighbor[1]
    nEdge3 = neighbor[2]
    nEdge4 = neighbor[3]

    if nEdge1 is not None:
        x1_1, y1_1 = nEdge1.shape[0]
        x2_1, y2_1 = nEdge1.shape[-1]
        theta1 = math.atan2(y2_1 - y1_1, x2_1 - x1_1)
        nEdge1Info = (x1_1, y1_1, theta1)
    else:
        nEdge1Info = None

    if nEdge2 is not None:
        x1_2, y1_2 = nEdge2.shape[0]
        x2_2, y2_2 = nEdge2.shape[-1]
        theta2 = math.atan2(y2_2 - y1_2, x2_2 - x1_2)
        nEdge2Info = (x1_2, y1_2, theta2)
    else:
        nEdge2Info = None

    if nEdge3 is not None:
        x1_3, y1_3 = nEdge3.shape[0]
        x2_3, y2_3 = nEdge3.shape[-1]
        theta3 = math.atan2(y2_3 - y1_3, x2_3 - x1_3)
        nEdge3Info = (x1_3, y1_3, theta3)
    else:
        nEdge3Info = None

    if nEdge4 is not None:
        x1_4, y1_4 = nEdge4.shape[0]
        x2_4, y2_4 = nEdge4.shape[-1]
        theta4 = math.atan2(y2_4 - y1_4, x2_4 - x1_4)
        nEdge4Info = (x1_4, y1_4, theta4)
    else:
        nEdge4Info = None

    nEdgeInfo = [nEdge1Info, nEdge2Info, nEdge3Info, nEdge4Info]

    euclidean_differences_info = []
    euclidean_differences_value = []
    ret = None

    for c_piece in candidate_pieces:
        minEuclidean = 999999999
        minIdx = 0
        minTheta = None
        minTrans = None

        for i in range(4):
            if ((nEdge1 is not None) and (c_piece.edges_[i].type == nEdge1.type)) or \
                    ((nEdge2 is not None) and (c_piece.edges_[(i+1) % 4].type == nEdge2.type)) or \
                    ((nEdge3 is not None) and (c_piece.edges_[(i+2) % 4].type == nEdge3.type)) or \
                    ((nEdge4 is not None) and (c_piece.edges_[(i+3) % 4].type == nEdge4.type)):
                continue
            else:
                total_euclidean = 0
                one_or_two = 1
                thetaTemp1 = None
                thetaTemp2 = None
                transTemp1 = None
                transTemp2 = None

                for idx, nEdge in enumerate(neighbor):
                    if nEdge is None:
                        continue

                    x, y, theta = nEdgeInfo[idx]
                    edge = c_piece.edges_[(i+idx) % 4]

                    translated_pixels1, theta_diff1, translate1 = rotate_edge_pixels(x, y, theta, edge, -1)
                    translated_pixels2, theta_diff2, translate2 = rotate_edge_pixels(x, y, theta, edge, 0)

                    diff1 = euclidean(nEdge.shape, np.flip(translated_pixels1, axis=0))
                    diff2 = euclidean(nEdge.shape, translated_pixels2)
                    total_euclidean += min(diff1, diff2)
                    thetaTemp1 = theta_diff1
                    thetaTemp2 = theta_diff2
                    transTemp1 = translate1
                    transTemp2 = translate2

                    if diff1 > diff2:
                        one_or_two = 2

                if total_euclidean < minEuclidean:
                    minEuclidean = total_euclidean
                    minIdx = i
                    if one_or_two == 1:
                        minTheta = thetaTemp1
                        minTrans = transTemp1
                    else:
                        minTheta = thetaTemp2
                        minTrans = transTemp2

        euclidean_differences_value.append(minEuclidean)
        euclidean_differences_info.append((c_piece, (minIdx+finalI) % 4, minTheta, minTrans))

    topIdx = np.argsort(euclidean_differences_value)[0:euclidean_num]
    min_ = 99999999
    for idx in topIdx:
        edge_idx = euclidean_differences_info[idx][1]
        edge = euclidean_differences_info[idx][0].edges_[edge_idx]
        color_norm = c_edge.color_norm(edge)
        if color_norm + euclidean_differences_value[idx] / 75 < min_:
            min_ = color_norm + euclidean_differences_value[idx] / 75
            ret = euclidean_differences_info[idx]

    return ret
