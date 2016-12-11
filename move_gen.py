from bb import *

def knight_moves(knights, own):
    for knight in iterate_pieces(knights):
        attacks = knight_attack(knight) & (own ^ FULL_BOARD)
        for attack in iterate_pieces(attacks):
            yield knight, attack

def bishop_moves(b, own, other):
    for bishop in iterate_pieces(b):
        yield from attacks_generic(bishop, own, other, bishop_attack)

def rook_moves(b, own, other):
    for rook in iterate_pieces(b):
        yield from attacks_generic(rook, own, other, rook_attack)

def queen_moves(b, own, other):
    for queen in iterate_pieces(b):
        yield from attacks_generic(queen, own, other, queen_attack)

def pawn_moves(pawns, own, other,
               side, last_move_piece,
               last_from_sq, last_to_sq):
    for pawn in iterate_pieces(pawns):
        en_pessant_squares = en_pessant_sq(side, last_move_piece, last_from_sq, last_to_sq)
        other |= en_pessant_squares
        captures = pawn_capture(pawn, side, other)
        non_capture_moves = pawn_move_non_capture(pawn, side, own, other)
        moves = captures | non_capture_moves
        for move in iterate_pieces(moves):
            yield (pawn, move)

def king_castle_moves(own, other, attacked, position_flags):
    if white_to_move(position_flags):
        if white_can_castle_kingside(position_flags, attacked, (own | other) ^ FULL_BOARD):
            yield 8, 2
        if white_can_castle_queenside(position_flags, attacked, (own | other) ^ FULL_BOARD):
            yield 8, 0x20
    else:
        if black_can_castle_kingside(position_flags, attacked, (own | other) ^ FULL_BOARD):
            yield 8 << 56, 2 << 56
        if black_can_castle_queenside(position_flags, attacked, (own | other) ^ FULL_BOARD):
            yield 8 << 56, 32 << 56

def king_moves(king, own, attacked):
    moves = king_attack(king) & (own ^ FULL_BOARD) & (attacked ^ FULL_BOARD)
    for move in iterate_pieces(moves):
        yield king, move

def attacks_generic(p, own, other, attack_fn):
    attacks = attack_fn(p, (own | other) ^ FULL_BOARD) & (own ^ FULL_BOARD)
    return ((p, to_sq) for to_sq in iterate_pieces(attacks))

def pawn_capture(pawn, side_to_move, other):
    return pawn_attack(pawn, side_to_move) & other
    
def pawn_move_non_capture(pawn, side_to_move, own, other):
    # b: pawn sq
    not_own = own ^ FULL_BOARD
    not_other = other ^ FULL_BOARD
    not_occupied = not_own & not_other
    if side_to_move == Side.WHITE:
        moves = (pawn << 8) & not_occupied
        # print('--')
        # print_bb(not_occupied & ((pawn << 8) | (pawn << 16)))
        # print('--end')
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

def am_in_check(attacks, king):
    return attacks & king > 0

def preserved_castle_rights(position_flags, side):
    if side == Side.WHITE:
        return not (position_flags & 1 or (position_flags & 12) == 12)
    return not (position_flags & 2 or (position_flags & 48) == 48)

def piece_attacks(piece_type, piece_squares, free):
    base_type = PieceType.base_type(piece_type)
    if base_type == PieceType.P:
        return pawn_attack(piece_squares, PieceType.get_side(piece_type))
    elif base_type == PieceType.N:
        return knight_attack(piece_squares)
    elif base_type == PieceType.B:
        return bishop_attack(piece_squares, free)
    elif base_type == PieceType.Q:
        return queen_attack(piece_squares, free)
    elif base_type == PieceType.K:
        return king_attack(piece_squares)
    elif base_type == PieceType.R:
        return rook_attack(piece_squares, free)

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
