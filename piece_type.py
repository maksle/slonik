from enum import Enum
from side import Side

class PieceType(Enum):
    NULL = 0
    P = PAWN = W_PAWN = 1
    N = KNIGHT = W_KNIGHT = 2
    B = BISHOP = W_BISHOP = 3
    R = ROOK = W_ROOK = 4
    Q = QUEEN = W_QUEEN = 5
    K = W_KING = 6
    B_PAWN = 7
    B_KNIGHT = 8
    B_BISHOP = 9
    B_ROOK = 10
    B_QUEEN = 11
    B_KING = 12

    @classmethod
    def base_type(cls, val):
        if val <= 6:
            return val
        else:
            return val - 6

    @classmethod
    def get_side(cls, piece_type):
        if cls.is_white(piece_type):
            return Side.WHITE.value
        else:
            return Side.BLACK.value
        
    @classmethod
    def is_white(cls, val):
        return val >= 1 and val <= 6
            
    @classmethod
    def is_black(cls, val):
        return val >= 7 and val <= 12

    @classmethod
    def piece(cls, piece, side):
        if side == Side.WHITE.value:
            return piece
        else:
            return piece + 6

    @classmethod
    def piece_types(cls, base_only=True, side=None):
        if base_only:
            return [PieceType.P.value, PieceType.N.value, PieceType.B.value,
                    PieceType.R.value, PieceType.Q.value, PieceType.K.value]
        else:
            if side is None:
                return [PieceType.W_PAWN.value, PieceType.W_KNIGHT.value, PieceType.W_BISHOP.value,
                        PieceType.W_ROOK.value, PieceType.W_QUEEN.value, PieceType.W_KING.value,
                        PieceType.B_PAWN.value, PieceType.B_KNIGHT.value, PieceType.B_BISHOP.value,
                        PieceType.B_ROOK.value, PieceType.B_QUEEN.value, PieceType.B_KING.value]
            elif side == Side.WHITE.value:
                return [PieceType.W_PAWN.value, PieceType.W_KNIGHT.value, PieceType.W_BISHOP.value,
                        PieceType.W_ROOK.value, PieceType.W_QUEEN.value, PieceType.W_KING.value]
            else:
                return [PieceType.B_PAWN.value, PieceType.B_KNIGHT.value, PieceType.B_BISHOP.value,
                        PieceType.B_ROOK.value, PieceType.B_QUEEN.value, PieceType.B_KING.value]

