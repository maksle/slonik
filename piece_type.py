# from enum import Enum
from side import Side

class PieceType():
    NULL = 0
    P = PAWN = W_PAWN = 1
    N = KNIGHT = W_KNIGHT = 2
    B = BISHOP = W_BISHOP = 3
    R = ROOK = W_ROOK = 4
    Q = QUEEN = W_QUEEN = 5
    K = KING = W_KING = 6
    BP = B_PAWN = 7
    BN = B_KNIGHT = 8
    BB = B_BISHOP = 9
    BR = B_ROOK = 10
    BQ = B_QUEEN = 11
    BK = B_KING = 12

    @classmethod
    def base_type(cls, val):
        if val <= 6:
            return val
        else:
            return val - 6

    @classmethod
    def get_side(cls, piece_type):
        if cls.is_white(piece_type):
            return Side.WHITE
        else:
            return Side.BLACK
        
    @classmethod
    def is_white(cls, val):
        return val >= 1 and val <= 6
            
    @classmethod
    def is_black(cls, val):
        return val >= 7 and val <= 12

    @classmethod
    def piece(cls, piece, side):
        if side == Side.WHITE:
            return piece
        else:
            return piece + 6

    @classmethod
    def piece_types(cls, base_only=True, side=None):
        if side is None:
            if base_only:
                return [PieceType.P, PieceType.N, PieceType.B,
                        PieceType.R, PieceType.Q, PieceType.K]
            else:
                return [PieceType.W_PAWN, PieceType.W_KNIGHT, PieceType.W_BISHOP,
                        PieceType.W_ROOK, PieceType.W_QUEEN, PieceType.W_KING,
                        PieceType.B_PAWN, PieceType.B_KNIGHT, PieceType.B_BISHOP,
                        PieceType.B_ROOK, PieceType.B_QUEEN, PieceType.B_KING]
        elif side == Side.WHITE:
            return [PieceType.W_PAWN, PieceType.W_KNIGHT, PieceType.W_BISHOP,
                    PieceType.W_ROOK, PieceType.W_QUEEN, PieceType.W_KING]
        else:
            return [PieceType.B_PAWN, PieceType.B_KNIGHT, PieceType.B_BISHOP,
                    PieceType.B_ROOK, PieceType.B_QUEEN, PieceType.B_KING]
