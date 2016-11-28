from material import MG_PIECES
from piece_type import PieceType as PT
from position import Position
from bb import *

def lowest_attacker(position, square, side=None):
    """Finds lowest piece of `side` attacking a square"""
    side = position.side_to_move() if side is None else side

    # pawn
    if side == Side.WHITE:
        possible_from_squares = south_west(square) | south_east(square)
    else:
        possible_from_squares = north_west(square) | north_east(square)
    piece_type = PieceType.piece(PieceType.P, side)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # knight
    piece_type = PieceType.piece(PieceType.N, side)
    possible_from_squares = knight_attack(square)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # sliders
    for piece_type in [PieceType.B, PieceType.R, PieceType.Q]:
        piece_type_stm = PieceType.piece(piece_type, side)
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
    piece_type = PieceType.piece(PieceType.K, side)
    possible_from_squares = king_attack(square)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, attackers

def next_en_prise(position, side, move=None):
    """finds next en-prise piece for `side` optionally after `move` is done"""
    # update the position
    if move:
        if move.position:
            pos = move.position
        else:
            pos = Position(position)
            pos.make_move(move)
    else:
        pos = position

    for pt in sorted(PieceType.piece_types(side=side), reverse=True):
        if PieceType.base_type(pt) == PieceType.P:
            continue
        for square in iterate_pieces(pos.pieces[pt]):
            lowest = lowest_attacker(pos, square, side ^ 1)
            if not lowest: continue
            attacker_pt, attacker_sq = lowest
            # equal val but undefended more readily handled by eval_see
            if MG_PIECES[PieceType.base_type(attacker_pt)] < MG_PIECES[PieceType.base_type(pt)] \
               or not position.attacks[side] & square: # not defended
                # TODO: re not defended, it's not accurate if the piece defending is pinned to a heavier piece
                yield pt, square, attacker_pt, attacker_sq
                break

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
    king_sq = position.pieces[PieceType.piece(PieceType.K, side)]
    attacks = king_attack(king_sq)
    if side == Side.WHITE:
        attacks |= attacks << 8
    else:
        attacks |= attacks >> 8
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
    pt = PieceType.piece(PieceType.P, side)
    pawns = position.pieces[pt]
    penalty = 0
    # doubled
    if side == Side.WHITE:
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
    pawn_type = PieceType.piece(PieceType.P, side)
    pawn_cover = king_zone & position.pieces[pawn_type]
    return count_bits(pawn_cover) * 40

def rook_position_bonus(rook, position, side):
    pawns_us = position.pieces[PieceType.piece(PieceType.P, side)]
    pawns_them = position.pieces[PieceType.piece(PieceType.P, side ^ 1)]
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
    pawns_us = position.pieces[PieceType.piece(PieceType.P, side)]
    bonus = 25
    if base_type == PieceType.N:
        bonus += 10
    if side == Side.WHITE and (south_west(minor) | south_east(minor)) & pawns_us:
        return bonus
    if side == Side.BLACK and (north_west(minor) | north_east(minor)) & pawns_us:
        return bonus
    return bonus
    
def mobility(position, side, pinned):
    """Bonus for legal moves not attacked by lower weight piece. Pinned pieces
    have restricted mobility"""
    mobility = 0
    piece_types = [PT.P, PT.N, PT.B, PT.R, PT.Q]

    pinned_piece_types = []
    if len(pinned):
        pinned_piece_types = [position.squares[bit_position(p)] for p in pinned]
    
    for base_pt in piece_types:
        pt = PT.piece(base_pt, side)
        
        # if attacked by lower weight piece, it doesn't count
        lower_wts = (pt for pt in piece_types if pt < base_pt)
        opp_attacks = 0
        for piece_type in lower_wts:
            opp_pt = PieceType.piece(piece_type, side ^ 1)
            opp_attacks |= position.piece_attacks[opp_pt]
        
        attacks = 0
        
        if len(pinned_piece_types) and pt in pinned_piece_types:
            occupied = position.occupied[Side.WHITE] | position.occupied[Side.BLACK]
            free = occupied ^ FULL_BOARD

            for p_sq in iterate_pieces(position.pieces[pt]):
                this_piece_attacks = 0
                if pt == PieceType.P:
                    this_piece_attacks = pawn_attack(p_sq, side)
                elif pt == PieceType.N:
                    this_piece_attacks = knight_attack(p_sq)
                elif pt == PieceType.B:
                    this_piece_attacks = bishop_attack(p_sq, free)
                elif pt == PieceType.R:
                    this_piece_attacks = rook_attack(p_sq, free)
                elif pt == PieceType.Q:
                    this_piece_attacks = queen_attack(p_sq, free)
                elif pt == PieceType.K:
                    this_piece_attacks = king_attack(p_sq)

                # mobility restricted for pinned pieces
                if p_sq in pinned:
                    k_sq = position.pieces[PieceType.piece(PieceType.K, side)]
                    this_piece_attacks &= LINE_SQS[bit_position(p_sq)][bit_position(k_sq)]
                
                attacks |= this_piece_attacks
        else:
            attacks = position.piece_attacks[pt]
        
        attacks &= opp_attacks ^ FULL_BOARD
        attacks &= position.occupied[side] ^ FULL_BOARD
        mobility += count_bits(attacks)

    return mobility

def attacked_pieces(position, side):
    return position.occupied[side] & (position.attacks[side] ^ FULL_BOARD)
    
def unprotected_penalty(position, side, pins):
    us = position.occupied[side]
    them = position.occupied[side ^ 1]
    free = (us | them) ^ FULL_BOARD
    us_attacked = attacked_pieces(position, side)
    penalty = 0
    for pt in PieceType.piece_types(side=side):
        num = count_bits(position.pieces[pt] & us_attacked)
        penalty += num * MG_PIECES[PieceType.P]
        
        defended = us_attacked & position.pieces[pt] & position.attacks[side]
        for defended_piece in iterate_pieces(defended):
            if defended_piece & position.piece_attacks[PieceType.piece(PieceType.P, side=side)]:
                # defended by pawn
                penalty -= MG_PIECES[PieceType.P] / 2
            else:
                penalty -= MG_PIECES[PieceType.P] / 4

        if PieceType.base_type(pt) == PieceType.P:
            continue
        # possible to get attack from pawn. Penalty regardless if defended
        for p in iterate_pieces(position.pieces[pt]):
            if side == Side.WHITE:
                pawn_attack_sqs = (north_east(p) | north_west(p)) & free
            else:
                pawn_attack_sqs = (south_east(p) | south_west(p)) & free

            for pawn_attack_sq in iterate_pieces(pawn_attack_sqs):
                if side == Side.WHITE:
                    pawn_from_sqs = (pawn_attack_sq << 8)
                    if pawn_attack_sq & RANKS[4]:
                        pawn_from_sqs |= (pawn_attack_sq << 16)
                    pawn_from_sqs &= free << 8
                else:
                    pawn_from_sqs = (pawn_attack_sq >> 8)
                    if pawn_attack_sq & RANKS[3]:
                        pawn_from_sqs |= (pawn_attack_sq >> 16)
                    pawn_from_sqs &= free >> 8
                
                pawn_from_sqs &= position.pieces[PieceType.piece(PieceType.P, side=side^1)]
                if pawn_from_sqs:
                    penalty += (MG_PIECES[PieceType.base_type(pt)] / 8) - 20
                    # more penalty if the piece is pinned
                    if p in pins:
                        penalty += (MG_PIECES[PieceType.base_type(pt)] / 3) - 20
                        # more penalty if the pawn is supported on the attack square
                        if position.attacks[side ^ 1] & pawn_attack_sq or \
                           position.attacks[side] & pawn_attack_sq == 0:
                            penalty += (MG_PIECES[PieceType.base_type(pt)] / 3) - 20
    
    return int(penalty * 1 / 4)

def discoveries_and_pins(position, side, target_piece_type=PieceType.K):
    """Return squares of singular pieces between sliders and the king of side
    `side`. Blockers of opposite side can move and cause discovered check, and
    blockers of same side are pinned. `target_piece_type` is piece things are pinned to."""
    pinned = []
    discoverers = []
    target_piece_sq = position.pieces[PieceType.piece(target_piece_type, side)]
    us = position.occupied[side]
    them = position.occupied[side ^ 1]
    all_occupied = us | them

    diags = PSEUDO_ATTACKS[PieceType.B][bit_position(target_piece_sq)]
    diag_sliders = position.pieces[PieceType.piece(PieceType.B, side ^ 1)]
    diag_sliders |= position.pieces[PieceType.piece(PieceType.Q, side ^ 1)]
    snipers = diags & diag_sliders

    cross = PSEUDO_ATTACKS[PieceType.R][bit_position(target_piece_sq)]
    cross_sliders = position.pieces[PieceType.piece(PieceType.R, side ^ 1)]
    cross_sliders |= position.pieces[PieceType.piece(PieceType.Q, side ^ 1)]
    snipers |= cross & cross_sliders
    
    for sniper_sq in iterate_pieces(snipers):
        squares_between = BETWEEN_SQS[bit_position(sniper_sq)][bit_position(target_piece_sq)]
        if count_bits(squares_between & all_occupied) == 1:
            p = us & squares_between
            d = them & squares_between
            if p: pinned.append(p)
            else: discoverers.append(d)

    return (discoverers, pinned)
