from Puzzle.PuzzlePiece import *
from Puzzle.Extractor import Extractor
from cv2 import cv2
import math
import numpy as np
from Puzzle.Enums import *
import sys
import scipy
from Puzzle.tuple_helper import *
import os

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

#================이미지 처리 부분==========================================================================================
def show(large_board, dx, dy, mx, my, img_count, path):
    large_board = np.roll(large_board, [-dx, -dy], axis=(0, 1))
    large_board = large_board[0:mx - dx + 100, 0:my - dy + 100]

    path = path.split('/')[2]
    filename = path.split('.')[0]
    extension = path.split('.')[1]
    if not os.path.exists("../result/{0}".format(filename)):
        os.makedirs("../result/{0}".format(filename))
    cv2.imwrite("../result/{0}/{0}_{1}.{2}".format(filename, img_count, extension) , large_board)

    if img_count == 'finish':
        make_mp4(filename, extension, count)

def make_mp4(filename, extension, total_count) :
    x, y, _ = cv2.imread("../result/{0}/{0}_finish.{1}".format(filename, extension)).shape
    size = (max(x, y)+50, max(x, y)+50)

    video_path = '../result/{0}/motion.avi'.format(filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 8, size)
    for i in range(0, total_count+1):
        img = cv2.imread("../result/{0}/{0}_{1}.{2}".format(filename, i, extension))

        top = (size[0]-img.shape[0])//2
        bottom = size[0]-img.shape[0] - top
        left = (size[1]-img.shape[1])//2
        right = size[1]-img.shape[1]-left
        resized_img = cv2.copyMakeBorder(img, top=top, bottom=bottom, left=left,
                                         right=right, borderType=cv2.BORDER_CONSTANT, value = [255, 255, 255])
        out.write(resized_img)
        if i == total_count :
            out.write(resized_img)
            out.write(resized_img)
            out.write(resized_img)
            out.write(resized_img)
    out.release()
    sys.exit()

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
#=======================================================================================================================

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


def get_candidate_by_length(ref_edge, pieces, candidate_num=10):
    diff = np.zeros((len(pieces), len(pieces[0].edges_)))
    for i, piece in enumerate(pieces):
        for j, edge in enumerate(piece.edges_):
            diff[i, j] = abs(ref_edge.length - edge.length)
    diff = np.sort(diff, axis=1)[:, 0]
    idx = np.argsort(diff).astype(int)[:candidate_num] \
        if candidate_num > len(pieces) else np.argsort(diff).astype(int)[:]
    return np.array(pieces)[idx]

#+==============================================================
#Updated to align the first piece
def get_info_align_to_axis(piece : PuzzlePiece):
    border_edge = None
    for edge in piece.edges_:
        if edge.type == TypeEdge.BORDER:
            border_edge = edge

    r_x1, r_y1 = 1,2
    r_x2, r_y2 = 1,1
    r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)
    _, theta, trans = rotate_edge_pixels(r_x1, r_y1, r_theta, edge=border_edge, task=0)
    return theta, trans

def rotate_translate_color(piece, theta, trans, largeBoard, boundaries = None):
    # rotate, translate and color
    rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    for edge in piece.edges_:
        edge.shape = edge.shape @ rot_matrix
        edge.shape = edge.shape + trans
    if boundaries is None :
        x_size, y_size, _ = largeBoard.shape
        minX, minY, maxX, maxY  = x_size -1, y_size-1, 0, 0
    else :
        (minX, minY, maxX, maxY) = boundaries
    for pixel in piece.img_piece_:
        pixel.pos = rot_matrix @ pixel.pos
        x = int(pixel.pos[0] + trans[1])
        y = int(pixel.pos[1] + trans[0])

        minX, minY, maxX, maxY = boundary(x, y, minX, minY, maxX, maxY)
        largeBoard[x][y] = pixel.color
    return minX, minY, maxX, maxY
#=========================================================================
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
        global count
        count = -1

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

            theta, trans = get_info_align_to_axis(b_piece)
            minX, minY, maxX, maxY = rotate_translate_color(b_piece, theta, trans, largeBoard)

            complete_pieces.append(b_piece)
            border_pieces.remove(b_piece)
            break

        # Border 조각 맞추기
        while len(border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:

                for c_edge in c_piece.edges_:
                    if c_piece.nBorders_ != 2 and find_opposite_side(c_piece, c_edge.direction) == TypeEdge.BORDER:
                        continue

                    if not c_edge.connected:
                        candidate_pieces = get_candidate_by_length(c_edge, border_pieces, len(border_pieces)//2+3)
                        friend, i, theta, trans = find_matching_piece(c_piece, c_edge, candidate_pieces, len(candidate_pieces)*2+3)
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
            count += 1
            show(largeBoard, minX, minY, maxX, maxY, count, path)

        # Border 조각 완료 후 connected 정리
        for piece in complete_pieces:
            for edge in piece.edges_:
                if not edge.connected:
                    if piece.nBorders_ == 2:
                        edge.connected = True
                    else:
                        if piece.edge_in_direction(rotate_direction(edge.direction, 2)).type != TypeEdge.BORDER:
                            edge.connected = True

        # Center 조각 맞추기
        while len(non_border_pieces) > 0:
            is_valid = True
            for c_piece in complete_pieces:
                for c_edge in c_piece.edges_:
                    num = 0
                    for dir in directions :
                        if add_tuple(add_tuple(c_piece.position, c_edge.direction.value), dir.value) in grid_completed:
                            num += 1
                    if num < 2:
                        continue
                    if not c_edge.connected:
                        candidate_pieces = get_candidate_by_length(c_edge, non_border_pieces,  len(non_border_pieces)//2+3)
                        #=======NEIGHBOR CHANGE==========
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
                        #=======NEIGHBOR CHANGE==========
                        neighbor = list(neighbors.values())
                        friend, i, theta, trans = find_matching_piece_center(neighbor, c_edge, candidate_pieces,  len(non_border_pieces)*2+3)
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
            count += 1
            show(largeBoard, minX, minY, maxX, maxY, count, path)

#========이미지 처리 부분==================================================================================================
        show(largeBoard, minX, minY, maxX, maxY, 'finish', path)
#=======================================================================================================================


def find_matching_piece(ref_piece, ref_edge, candidate_pieces, euclidean_num):
    r_x1, r_y1 = ref_edge.shape[0]
    r_x2, r_y2 = ref_edge.shape[-1]
    r_theta = math.atan2(r_y2 - r_y1, r_x2 - r_x1)

    euclidean_differences_info = []
    euclidean_differences_value = []

    for candidate_piece in candidate_pieces:
        for i, edge in enumerate(candidate_piece.edges_):
            if not edge.is_compatible(ref_edge):
                continue
            #===================Rotation 1=======================
            translated_pixels1, theta_diff1, translate1 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, -1)
            #=====================Rotation 2======================
            translated_pixels2, theta_diff2, translate2 = rotate_edge_pixels(r_x1, r_y1, r_theta, edge, 0)
            #======================Euclidean====================
            diff1 = euclidean(ref_edge.shape, np.flip(translated_pixels1, axis=0))
            diff2 = euclidean(ref_edge.shape, translated_pixels2)
            rollback_direction = rotate_direction(edge.direction, 2)
            #=====================1, 2 통합======================
            if diff2 < diff1:
                translated_pixels1 = translated_pixels2
                theta_diff1 = theta_diff2
                translate1 = translate2
                diff1 = diff2

            update_dir(ref_edge.direction, candidate_piece, i)
            if (ref_piece.nBorders_ !=2 and ref_piece.is_border_aligned(candidate_piece)):
                _, theta_diff, _ = ref_piece.border_angle(candidate_piece, task =0)
                _, border_theta_diff = divmod(theta_diff - theta_diff1, np.pi)
                if abs(border_theta_diff)<0.05 or np.pi - abs(border_theta_diff)< 0.05:
                    euclidean_differences_value.append(diff1)
                    euclidean_differences_info.append((candidate_piece, i, theta_diff1, translate1))
            elif  ref_piece.nBorders_== 2 :
                euclidean_differences_value.append(diff1)
                euclidean_differences_info.append((candidate_piece, i, theta_diff1, translate1))

            update_dir(rollback_direction,  candidate_piece, i)
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

#=======================================================================================================================


def find_matching_piece_center(neighbor, c_edge, candidate_pieces, euclidean_num):
    neighbor_num = 0
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
        neighbor_num += 1
        x1_1, y1_1 = nEdge1.shape[0]
        x2_1, y2_1 = nEdge1.shape[-1]
        theta1 = math.atan2(y2_1 - y1_1, x2_1 - x1_1)
        nEdge1Info = (x1_1, y1_1, theta1)
    else:
        nEdge1Info = None

    if nEdge2 is not None:
        neighbor_num += 1
        x1_2, y1_2 = nEdge2.shape[0]
        x2_2, y2_2 = nEdge2.shape[-1]
        theta2 = math.atan2(y2_2 - y1_2, x2_2 - x1_2)
        nEdge2Info = (x1_2, y1_2, theta2)
    else:
        nEdge2Info = None

    if nEdge3 is not None:
        neighbor_num += 1
        x1_3, y1_3 = nEdge3.shape[0]
        x2_3, y2_3 = nEdge3.shape[-1]
        theta3 = math.atan2(y2_3 - y1_3, x2_3 - x1_3)
        nEdge3Info = (x1_3, y1_3, theta3)
    else:
        nEdge3Info = None

    if nEdge4 is not None:
        neighbor_num += 1
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

    partner_info_per_piece = []

    for c_piece in candidate_pieces:

        for i in range(4):
            if ((nEdge1 is not None) and (c_piece.edges_[i].type == nEdge1.type)) or \
                    ((nEdge2 is not None) and (c_piece.edges_[(i+1) % 4].type == nEdge2.type)) or \
                    ((nEdge3 is not None) and (c_piece.edges_[(i+2) % 4].type == nEdge3.type)) or \
                    ((nEdge4 is not None) and (c_piece.edges_[(i+3) % 4].type == nEdge4.type)):
                continue
            else:
                total_euclidean = 0
                partner_info_temp = []

                for idx, nEdge in enumerate(neighbor):
                    if nEdge is None:
                        continue

                    x, y, theta = nEdgeInfo[idx]
                    edge = c_piece.edges_[(i+idx) % 4]
                    partner_info_temp.append((nEdge, edge))

                    translated_pixels1, theta_diff1, translate1 = rotate_edge_pixels(x, y, theta, edge, -1)
                    translated_pixels2, theta_diff2, translate2 = rotate_edge_pixels(x, y, theta, edge, 0)

                    diff1 = euclidean(nEdge.shape, np.flip(translated_pixels1, axis=0))
                    diff2 = euclidean(nEdge.shape, translated_pixels2)

                    if diff2 < diff1 :
                        diff1 = diff2
                        translated_pixels1, theta_diff1, translate1 = translated_pixels2, theta_diff2, translate2

                    total_euclidean += diff1
                    thetaTemp = theta_diff1
                    transTemp = translate1

                    euclidean_differences_value.append(total_euclidean)
                    euclidean_differences_info.append((c_piece, (i + finalI) % 4, thetaTemp, transTemp))
                    partner_info_per_piece.append(partner_info_temp)

    topIdx = np.argsort(euclidean_differences_value)[0:euclidean_num]
    min_ = 99999999
    for idx in topIdx:
        color_norm_total = 0
        pairs = partner_info_per_piece[idx]
        for n_edge, f_edge in pairs :
            color_norm_total += n_edge.color_norm(f_edge)

        if len(pairs) != 0:
            color_norm_total /= len(pairs)

        if color_norm_total + euclidean_differences_value[idx] / 75 < min_:
            min_ = color_norm_total + euclidean_differences_value[idx] / 75
            ret = euclidean_differences_info[idx]

    return ret
