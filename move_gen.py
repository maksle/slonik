import pyximport
pyximport.install()
from bb import *

def knight_moves(knights, own):
    for knight in iterate_pieces(knights):
        attacks = knight_attack(knight) & (own ^ FULL_BOARD)
        for attack in iterate_pieces(attacks):
            yield knight, attack

def bishop_moves(b, own, other):
    for bishop in iterate_pieces(b):
        moves = piece_attack(Pt.B, bishop, own|other) & (own ^ FULL_BOARD)
        for move in iterate_pieces(moves):
            yield (bishop, move)

def rook_moves(b, own, other):
    for rook in iterate_pieces(b):
        moves = piece_attack(Pt.R, rook, own|other) & (own ^ FULL_BOARD)
        for move in iterate_pieces(moves):
            yield (rook, move)
        
def queen_moves(b, own, other):
    for queen in iterate_pieces(b):
        moves = piece_attack(Pt.Q, queen, own|other) & (own ^ FULL_BOARD)
        for move in iterate_pieces(moves):
            yield (queen, move)

def pawn_moves(pawns, own, other, en_pessant_sq, side):
    for pawn in iterate_pieces(pawns):
        other |= en_pessant_sq or 0
        captures = pawn_capture(pawn, side, other)
        non_capture_moves = pawn_move_non_capture(pawn, side, own, other)
        moves = captures | non_capture_moves
        for move in iterate_pieces(moves):
            yield (pawn, move)

def pseudo_king_moves(position):
    side = position.side_to_move()
    ksq = position.pieces[PieceType.piece(PieceType.K, side)]
    flags = position.position_flags
    free = invert(position.occupied[Side.WHITE] | position.occupied[Side.BLACK])
    for sq in iterate_pieces(king_attack(ksq) & invert(position.occupied[side])):
        yield ksq, sq
    if position.white_to_move():
        if flags & 5 == 0 and (free & 6) == 6: # kingside 
            yield 8, 2
        if flags & 9 == 0 and (free & 0x70) == 0x70: # queenside 
            yield 8, 0x20
    else:
        if flags & 0x12 == 0 and (free & (6 << 56)) == (6 << 56): # kingside 
            yield 8 << 56, 2 << 56
        if flags & 0x22 == 0 and (free & (0x70 << 56)) == (0x70 << 56): # queenside 
            yield 8 << 56, 0x20 << 56

def pawn_capture(pawn, side_to_move, other):
    return pawn_attack(pawn, side_to_move) & other
    
def pawn_move_non_capture(pawn, side_to_move, own, other):
    # b: pawn sq
    not_own = own ^ FULL_BOARD
    not_other = other ^ FULL_BOARD
    not_occupied = not_own & not_other
    if side_to_move == Side.WHITE:
        moves = (pawn << 8) & not_occupied
        squares = ((pawn << 8) | (pawn << 16))
        if (pawn & 0xff00) > 0 \
           and (not_occupied & squares) == squares:
            moves |= pawn << 16
    elif side_to_move == Side.BLACK:
        moves = (pawn >> 8) & not_own & not_other
        squares = ((pawn >> 8) | (pawn >> 16))
        if (pawn & 0xff000000000000) > 0 \
           and (not_occupied & squares) == squares:
            moves |= pawn >> 16
    return moves

def is_capture(move, other):
    return (move & other) > 0

def preserved_kingside_castle_rights(position_flags, side):
    if side == Side.W:
        return position_flags & 5 == 0
    else:
        return position_flags & 18 == 0

def preserved_queenside_castle_rights(position_flags, side):
    if side == Side.W:
        return position_flags & 9 == 0
    else:
        return position_flags & 34 == 0

# Position flags:
# white king has moved - bit 1
# black king has moved - bit 2
# white rook kingside has moved - bit 3
# white rook queenside has moved - bit 4
# black rook kingside has moved - bit 5
# black rook queenside has moved - bit 6
# side to move - bit 7
def white_can_castle_kingside(position_flags, attacked, free):
    return position_flags & 5 == 0 \
        and (attacked & 0xe) == 0 \
        and (free & 6) == 6

def white_can_castle_queenside(position_flags, attacked, free):
    return position_flags & 9 == 0 \
        and (attacked & 0x38) == 0 \
        and (free & 0x70) == 0x70

def black_can_castle_kingside(position_flags, attacked, free):
    return position_flags & 0x12 == 0 \
        and (attacked & (0xe << 56)) == 0 \
        and (free & (6 << 56)) == (6 << 56)

def black_can_castle_queenside(position_flags, attacked, free):
    return position_flags & 0x22 == 0 \
        and (attacked & (0x38 << 56)) == 0 \
        and (free & (0x70 << 56)) == (0x70 << 56)

def side_to_move(position_flags):
    return (position_flags & 0x40) >> 6

def white_to_move(position_flags):
    return side_to_move(position_flags) == Side.WHITE

def black_to_move(position_flags):
    return side_to_move(position_flags) == Side.BLACK
