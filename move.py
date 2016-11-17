from print_bb import *
from piece_type import *
from bb import *

class Move():
    def __init__(self, piece_type, from_sq, to_sq, promo_piece = None):
        self.piece_type = piece_type
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.promo_piece = promo_piece
    
    def __str__(self):
        if self.piece_type is None or self.piece_type == PieceType.NULL.value:
            return ""
        return HUMAN_PIECE[self.piece_type] \
            + HUMAN_BOARD_INV[self.from_sq] \
            + "-" + HUMAN_BOARD_INV[self.to_sq]
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.piece_type == other.piece_type \
                and self.from_sq == other.from_sq \
                and self.to_sq == other.to_sq \
                and self.promo_piece == other.promo_piece
        else:
            return False

    def compact(self):
        # bits 0-5 from_sq
        # bits 6-11 to_sq
        # bits 12-15 piece_type
        # bits 16-19 promo_piece
        if self.piece_type == PieceType.NULL.value:
            return 0
        from_sq = len(bin(self.from_sq))-3
        to_sq = (len(bin(self.to_sq))-3) << 6
        piece_type = self.piece_type << 12
        promo_piece = (self.promo_piece or 0) << 16
        return from_sq | to_sq | piece_type | promo_piece
        
    @classmethod
    def move_uncompacted(cls, compacted):
        from_sq = 1 << (compacted & 0x3f)
        to_sq = 1 << ((compacted >> 6) & 0x3f)
        piece_type = (compacted >> 12) & 0xf
        promo_piece = (compacted >> 16) & 0xf
        return cls(piece_type, from_sq, to_sq, promo_piece or None)
        
    @classmethod
    def map_str_moves(cls, moves_list):
        return ' '.join(map(str,moves_list))
