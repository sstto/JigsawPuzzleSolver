from enum import Enum


class Directions(Enum):

    N = (0, 1)
    S = (0, -1)
    E = (1, 0)
    W = (-1, 0)


directions = [Directions.N, Directions.E, Directions.S, Directions.W]


def rotate_direction(dir, step):
    """ Find the clockwise next direction """

    i = directions.index(dir)
    return directions[(i + step) % 4]


class TypeEdge(Enum):
    """ Enum used to keep track of the type of edges """

    HOLE = 0
    HEAD = 1
    BORDER = 2
    UNDEFINED = 3


class TypePiece(Enum):
    """ Enum used to keep track of the type of pieces """

    CENTER = 0
    BORDER = 1
    ANGLE = 2
