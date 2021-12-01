import numpy as np

from Puzzle.Enums import directions, Directions, TypeEdge, TypePiece, rotate_direction

class PuzzlePiece():
    """
        Wrapper used to store informations about pieces of the puzzle.
        Contains the position of the piece in the puzzle graph, a list of edges,
        the list of pixels composing the piece, the number of borders and the type
        of the piece.
    """

    def __init__(self, edges, img_piece):
        self.position = (0, 0)
        self.edges_ = edges
        self.img_piece_ = img_piece  # List of Pixels
        self.img_piece_matrix = self.get_image_matrix()
        self.nBorders_ = self.number_of_border()
        self.type = TypePiece(self.nBorders_)


    def get_image_matrix(self):
        pixels_pos_color = []
        for img_piece in self.img_piece_:
            pixels_pos_color.append(np.concatenate((np.asarray(img_piece.pos),np.array(img_piece.color))))
        return np.array(pixels_pos_color)

    def number_of_border(self):
        """ Fast computations of the nunmber of borders """

        return len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))

    def rotate_edges(self, r):
        """ Rotate the edges """

        for e in self.edges_:
            e.direction = rotate_direction(e.direction, r)

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