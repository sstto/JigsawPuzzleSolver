from Puzzle.Distance import diff_match_edges, real_edge_compute, generated_edge_compute
from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2

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