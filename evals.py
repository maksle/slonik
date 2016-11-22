from material import MG_PIECES
from piece_type import PieceType as PT
from position import Position
from bb import *

def lowest_attacker(position, square):
    """Finds lowest-value piece attacking a square"""
    side = position.side_to_move()

    # pawn
    if position.white_to_move():
        possible_from_squares = south_west(square) | south_east(square)
    else:
        possible_from_squares = north_west(square) | north_east(square)
    piece_type = PieceType.piece(PieceType.P.value, side)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # knight
    piece_type = PieceType.piece(PieceType.N.value, side)
    possible_from_squares = knight_attack(square)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # sliders
    for piece_type in [PieceType.B, PieceType.R, PieceType.Q]:
        piece_type_stm = PieceType.piece(piece_type.value, side)
        if piece_type == PieceType.B:
            slider_attack = bishop_attack
        elif piece_type == PieceType.R:
            slider_attack = rook_attack
        else:
            slider_attack = queen_attack
        for attacker in iterate_pieces(position.pieces[piece_type_stm]):
            free = (position.occupied[side] | position.occupied[side^1]) ^ FULL_BOARD
            attacks = slider_attack(attacker, free)
            if attacks & square:
                return piece_type_stm, attacker

    # king
    piece_type = PieceType.piece(PieceType.K.value, side)
    possible_from_squares = king_attack(square)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, attackers

def eval_see(pos, move):
    position = Position(pos)
    balance = 0
    gains = []
    
    side = position.side_to_move()
    square = move.to_sq
    captured_piece_type = position.squares[len(bin(square))-3]
    if captured_piece_type:
        balance += MG_PIECES[PieceType.base_type(captured_piece_type)]
    position.make_move(move)
    
    gains.append(balance)

    while True:
        lowest = lowest_attacker(position, square)
        if not lowest:
            break
        piece_type_stm, attacker = lowest

        captured_piece_type = position.squares[len(bin(square))-3]
        balance = MG_PIECES[PieceType.base_type(captured_piece_type)] - gains[-1]
        gains.append(balance)

        if max(gains[-1], -gains[-2]) < 0:
            break

        # update the position
        position.pieces[piece_type_stm] ^= attacker ^ square
        position.squares[len(bin(square))-3] = piece_type_stm
        position.pieces[captured_piece_type] ^= square
        position.toggle_side_to_move()

    for i in reversed(range(1, len(gains))):
        gains[i-1] = -max(gains[i], -gains[i-1])

    move.see_score = gains[0]
    return gains[0]
    
def king_safety_squares(position, side):
    """squares surrounding king plus one rank further"""
    king_sq = position.pieces[PieceType.piece(PieceType.K.value, side)]
    attacks = king_attack(king_sq)
    if side == Side.WHITE.value:
        attacks |= attacks >> 8
    else:
        attacks |= attacks << 8
    return attacks

def king_zone_attack_bonus(king_zone, position, side):
    """Attackers of enemy king zone, each weighted by piece weight, including
xray attacks."""
    # TODO: finish implementing this
    bonus = 0
    types = 0
    for pt in PieceType.piece_types(side=side):
        num = count_bits(position.piece_attacks[pt] & king_zone)
        if num:
            types += 1
            bonus += num * 5 * types
    return bonus

def pawn_structure(position, side):
    pt = PieceType.piece(PieceType.P.value, side)
    pawns = position.pieces[pt]
    penalty = 0
    # doubled
    if side == Side.WHITE.value:
        penalty += count_bits(pawns & pawns << 8)
        penalty += count_bits(pawns & pawns << 16)
        penalty += count_bits(pawns & pawns << 24)
        penalty *= -10
    else:
        penalty += count_bits(pawns & pawns >> 8)
        penalty += count_bits(pawns & pawns >> 16)
        penalty += count_bits(pawns & pawns >> 24)
        penalty *= -10
    return penalty
    
def pawn_cover_bonus(king_zone, position, side):
    pawn_type = PieceType.piece(PieceType.P.value, side)
    pawn_cover = king_zone & position.pieces[pawn_type]
    pawn_value = MG_PIECES[PieceType.P.value]
    return count_bits(pawn_cover) * 20

def rook_position_bonus(rook, position, side):
    pawns_us = position.pieces[PieceType.piece(PieceType.P.value, side)]
    pawns_them = position.pieces[PieceType.piece(PieceType.P.value, side ^ 1)]
    bonus = 0
    fill = south_attack(rook, FULL_BOARD) | north_attack(rook, FULL_BOARD)
    if pawns_us & fill and not pawns_them & fill:
        # rook supporting passed pawn bonus
        bonus += 5
    if not pawns_us & fill:
        if pawns_them & fill:
            # semi-open file
            bonus += 20
        else:
            # open file
            bonus += 45

    rank_fill = east_attack(rook, FULL_BOARD) | west_attack(rook, FULL_BOARD)
    bonus += count_bits(rank_fill & pawns_them) * 70
    
    return bonus

def minor_outpost_bonus(minor, position, side):
    base_type = PieceType.base_type(minor)
    pawns_us = position.pieces[PieceType.piece(PieceType.P.value, side)]
    bonus = 25
    if base_type == PieceType.N.value:
        bonus += 10
    if side == Side.WHITE.value and (south_west(minor) | south_east(minor)) & pawns_us:
        return bonus
    if side == Side.BLACK.value and (north_west(minor) | north_east(minor)) & pawns_us:
        return bonus
    return bonus
    
def mobility(position, side):
    mobility = 0
    piece_types = [PT.P, PT.N, PT.B, PT.R, PT.Q]
    for base_pt in piece_types:
        pt = PT.piece(base_pt.value, side)
        
        # if attacked by lower weight piece, it doesn't count
        lower_wts = (pt.value for pt in piece_types if pt.value < base_pt.value)
        opp_attacks = 0
        for piece_type in lower_wts:
            opp_pt = PieceType.piece(piece_type, side ^ 1)
            opp_attacks |= position.piece_attacks[opp_pt]

        attacks = position.piece_attacks[pt]
        attacks &= opp_attacks ^ FULL_BOARD
        attacks &= position.occupied[side] ^ FULL_BOARD
        mobility += count_bits(attacks)
    return mobility

def unprotected(position, side):
    return position.occupied[side] & (position.attacks[side] ^ FULL_BOARD)
    
def unprotected_penalty(position, side):
    unprotected_pieces = unprotected(position, side)
    penalty = 0
    for pt in PieceType.piece_types(side=side):
        num = count_bits(position.pieces[pt] & unprotected_pieces)
        penalty += num * MG_PIECES[PieceType.base_type(pt)]
    return penalty // 200
