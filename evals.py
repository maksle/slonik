import pprint
from collections import defaultdict
from material import MG_PIECES, piece_counts, material_eval
from psqt import psqt_value
from move_gen import *
from piece_type import PieceType as Pt
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
    piece_type = Pt.piece(Pt.P, side)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # knight
    piece_type = Pt.piece(Pt.N, side)
    possible_from_squares = knight_attack(square)
    attackers = position.pieces[piece_type] & possible_from_squares
    if attackers:
        return piece_type, ls1b(attackers)

    # sliders
    for piece_type in [Pt.B, Pt.R, Pt.Q]:
        piece_type_stm = Pt.piece(piece_type, side)
        if piece_type == Pt.B:
            slider_attack = bishop_attack
        elif piece_type == Pt.R:
            slider_attack = rook_attack
        else:
            slider_attack = queen_attack
        for attacker in iterate_pieces(position.pieces[piece_type_stm]):
            free = (position.occupied[side] | position.occupied[side^1]) ^ FULL_BOARD
            attacks = slider_attack(attacker, free)
            if attacks & square:
                return piece_type_stm, attacker

    # king
    piece_type = Pt.piece(Pt.K, side)
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

    for pt in sorted(Pt.piece_types(side=side), reverse=True):
        if Pt.base_type(pt) == Pt.P:
            continue
        for square in iterate_pieces(pos.pieces[pt]):
            lowest = lowest_attacker(pos, square, side ^ 1)
            if not lowest: continue
            attacker_pt, attacker_sq = lowest
            # equal val but undefended more readily handled by eval_see
            if MG_PIECES[Pt.base_type(attacker_pt)] < MG_PIECES[Pt.base_type(pt)] \
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
        balance += MG_PIECES[Pt.base_type(captured_piece_type)]
    position.make_move(move)
    
    gains.append(balance)

    while True:
        lowest = lowest_attacker(position, square)
        if not lowest:
            break
        piece_type_stm, attacker = lowest

        captured_piece_type = position.squares[len(bin(square))-3]
        balance = MG_PIECES[Pt.base_type(captured_piece_type)] - gains[-1]
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
    king_sq = position.pieces[Pt.piece(Pt.K, side)]
    attacks = king_attack(king_sq)
    if side == Side.WHITE:
        attacks |= attacks << 8
    else:
        attacks |= attacks >> 8
    return attacks

def king_zone_attack_bonus(king_zone, position, side):
    """Attackers of enemy king zone, each weighted by piece weight, including
xray attacks."""
    us = side
    them = side ^ 1
    bonus = 0

    # b = position.piece_attacks[Pt.piece(Pt.K, us)] 
    
    types = 0
    for pt in Pt.piece_types(side=side):
        num = count_bits(position.piece_attacks[pt] & king_zone)
        if num:
            types += 1
            bonus += num * 10 * types

    
    return bonus

def pawn_structure(position, side):
    pt = Pt.piece(Pt.P, side)
    pawns = position.pieces[pt]

    penalty = 0

    # doubled and tripled pawns
    double_triple = 0
    if side == Side.WHITE:
        double_triple += count_bits(pawns & pawns << 8)
        double_triple += count_bits(pawns & pawns << 16)
        double_triple += count_bits(pawns & pawns << 24)
    else:
        double_triple += count_bits(pawns & pawns >> 8)
        double_triple += count_bits(pawns & pawns >> 16)
        double_triple += count_bits(pawns & pawns >> 24)
    penalty -= (double_triple * 30)
    
    return penalty
    
def pawn_cover_bonus(king_zone, position, side):
    pawn_type = Pt.piece(Pt.P, side)
    pawn_cover = king_zone & position.pieces[pawn_type]
    # increase this too much, king will move to get close to e4,d4 pawns! or only do this when king is castled?
    return count_bits(pawn_cover) * 6

def rook_position_bonus(rook, position, side):
    pawns_us = position.pieces[Pt.piece(Pt.P, side)]
    pawns_them = position.pieces[Pt.piece(Pt.P, side ^ 1)]
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
    bonus += count_bits(rank_fill & pawns_them) * 8

    # connected rooks, gets bonused once for each rook so halving it
    if position.piece_attacks[Pt.piece(Pt.R, side)] & rook:
        bonus += 15
        
    return bonus

def minor_outpost_bonus(minor, position, side, potentials):
    if minor == 0: return 0
    potential_them = potentials[side ^ 1]
    base_type = Pt.base_type(minor)
    pawns_us = position.pieces[Pt.piece(Pt.P, side)]
    outpost_ranks = [RANKS[3] | RANKS[4] | RANKS[5],
                     RANKS[2] | RANKS[3] | RANKS[4]]
    outpost_squares = outpost_ranks[side] & (potential_them ^ FULL_BOARD)
    if outpost_squares:
        # reachable squares
        if base_type == Pt.N: bonus = 25
        else: bonus = 15

        # successfully outposted
        if minor & outpost_squares: bonus += 10
        else: bonus += 5

        return bonus
    return 0

def bad_bishop_penalty(minor, position, side):
    color = minor & DARK_SQUARES
    pawns_us = position.pieces[Pt.piece(Pt.P, side)]
    if color == 0: pawns = pawns_us & WHITE_SQUARES
    else: pawns = pawns_us & DARK_SQUARES
    return count_bits(pawns) * 8
    
def minor_behind_pawn(minors, position, side):
    pawns_in_front = position.pieces[Pt.piece(Pt.P, side)] & shift_north(minors, side)
    pawns_in_front &= (RANKS[2] | RANKS[3] | RANKS[4] | RANKS[5])
    return count_bits(pawns_in_front) * 8
    
def mobility(position, side, pinned):
    """Bonus for legal moves not attacked by lower weight piece. Pinned pieces
    have restricted mobility"""
    mobility = 0
    piece_types = [Pt.P, Pt.N, Pt.B, Pt.R, Pt.Q]

    pinned_piece_types = []
    if len(pinned):
        pinned_piece_types = [position.squares[bit_position(p)] for p in pinned]
    
    for base_pt in piece_types:
        pt = Pt.piece(base_pt, side)
        
        # if attacked by lower weight piece, it doesn't count
        lower_wts = (pt for pt in piece_types if pt < base_pt)
        opp_attacks = 0
        for piece_type in lower_wts:
            opp_pt = Pt.piece(piece_type, side ^ 1)
            opp_attacks |= position.piece_attacks[opp_pt]
        
        attacks = 0
        
        if len(pinned_piece_types) and pt in pinned_piece_types:
            occupied = position.occupied[Side.WHITE] | position.occupied[Side.BLACK]
            free = occupied ^ FULL_BOARD

            for p_sq in iterate_pieces(position.pieces[pt]):
                this_piece_attacks = 0
                if pt == Pt.P:
                    this_piece_attacks = pawn_attack(p_sq, side)
                elif pt == Pt.N:
                    this_piece_attacks = knight_attack(p_sq)
                elif pt == Pt.B:
                    this_piece_attacks = bishop_attack(p_sq, free)
                elif pt == Pt.R:
                    this_piece_attacks = rook_attack(p_sq, free)
                elif pt == Pt.Q:
                    this_piece_attacks = queen_attack(p_sq, free)
                elif pt == Pt.K:
                    this_piece_attacks = king_attack(p_sq)

                # mobility restricted for pinned pieces
                if p_sq in pinned:
                    k_sq = position.pieces[Pt.piece(Pt.K, side)]
                    this_piece_attacks &= LINE_SQS[bit_position(p_sq)][bit_position(k_sq)]
                
                attacks |= this_piece_attacks
        else:
            attacks = position.piece_attacks[pt]
        
        attacks &= opp_attacks ^ FULL_BOARD
        attacks &= position.occupied[side] ^ FULL_BOARD
        mobility += count_bits(attacks)
        
    return mobility * 5

def attacked_pieces(position, side):
    return position.occupied[side] & position.attacks[side ^ 1]
    
def threats(position, side, pins):
    occupied = position.occupied[Side.WHITE] | position.occupied[Side.BLACK]
    free = occupied ^ FULL_BOARD
    rank2 = RANKS[1] if side == Side.WHITE else RANKS[6]
 
    bonus = 0
    penalty = 0
    # them_no_qk = position.occupied[side ^ 1] \
    #     ^ position.pieces[Pt.piece(Pt.Q, side ^ 1)] \
    #     ^ position.pieces[Pt.piece(Pt.K, side ^ 1)]
    # loose_pieces = them_no_qk & ((position.attacks[side] | position.attacks[side]) ^ FULL_BOARD)
    # if loose_pieces:
    #     bonus += 25

    # The following is copied as in stockfish:
    
    # non-pawn enemies attacked by pawn
    weak = (position.occupied[side ^ 1] ^ position.pieces[Pt.piece(Pt.P, side ^ 1)]) \
           & (position.piece_attacks[Pt.piece(Pt.P, side)])

    if weak:
        # our pawns protected by us or not attacked by them
        b = position.pieces[Pt.piece(Pt.P, side)] & (position.attacks[side] | (position.attacks[side] ^ FULL_BOARD))

        safe_threats = (shift_ne(b, side) | shift_nw(b, side)) & weak

        if weak ^ safe_threats:
            bonus += 70

        for threatened_piece in iterate_pieces(safe_threats):
            bonus += 150
            if Pt.base_type(position.squares[bit_position(threatened_piece)]) in [Pt.R, Pt.Q]:
                bonus += 50

    # non-pawn enemies defended by pawn
    defended = (position.occupied[side ^ 1] ^ position.pieces[Pt.piece(Pt.P, side ^ 1)]) \
           & (position.piece_attacks[Pt.piece(Pt.P, side ^ 1)])

    # enemies not defended by a pawn and under our attack
    weak = position.occupied[side ^ 1] \
           & (position.piece_attacks[Pt.piece(Pt.P, side ^ 1)] ^ FULL_BOARD) \
           & position.attacks[side]

    if defended | weak:
        # minor attacks
        minor_attack = position.piece_attacks[Pt.piece(Pt.N, side)] | position.piece_attacks[Pt.piece(Pt.B, side)]
        b = (defended | weak) & minor_attack
        for attacked in iterate_pieces(b):
            if position.squares[bit_position(attacked)] == Pt.N:
                bonus += 10
            if position.squares[bit_position(attacked)] > Pt.N:
                bonus += 45
            if position.squares[bit_position(attacked)] == Pt.Q:
                bonus += 30

        # rook attacks
        b = (position.pieces[Pt.piece(Pt.Q, side ^ 1)] | weak) & position.piece_attacks[Pt.piece(Pt.R, side)]
        for attacked in iterate_pieces(b):
            if position.squares[attacked] > Pt.P and position.squares[attacked] != Pt.R:
                bonus += 40

        # hanging
        bonus += 45 * count_bits(weak & (position.attacks[side ^ 1] ^ FULL_BOARD))

        # king attacks
        b = weak & position.piece_attacks[Pt.piece(Pt.K, side)]
        more_than_one = reset_ls1b(b) > 0
        if more_than_one: bonus += 9 # 120 for endgame
        elif b: bonus += 3 # 60 for endgame

    # bonus for pawn push that attacks pieces
    b = position.pieces[Pt.piece(Pt.P, side)]
    b = shift_north(b | (shift_north(b & rank2, side) & free), side)
    b &= free & (position.attacks[side] | invert(position.attacks[side ^ 1]))
    b = (shift_ne(b, side) | shift_nw(b, side)) & invert(position.piece_attacks[Pt.piece(Pt.P, side)])
    b2 = 0
    for p in pins:
        b2 |= p
    bonus += count_bits(b & b2) * 140 + count_bits(b & invert(b2)) * 40
    return bonus
        
def unprotected_penalty(position, side, pins):
    us = position.occupied[side]
    them = position.occupied[side ^ 1]
    free = (us | them) ^ FULL_BOARD
    us_attacked = attacked_pieces(position, side)
    penalty = 0
    for pt in Pt.piece_types(side=side):
        num = count_bits(position.pieces[pt] & us_attacked)
        penalty += num * 10
        
        defended = us_attacked & position.pieces[pt] & position.attacks[side]
        for defended_piece in iterate_pieces(defended):
            if defended_piece & position.piece_attacks[Pt.piece(Pt.P, side=side)]:
                # defended by pawn
                penalty -= MG_PIECES[Pt.P] * .25
            else:
                penalty -= MG_PIECES[Pt.P] * .125

        if Pt.base_type(pt) == Pt.P:
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
                
                pawn_from_sqs &= position.pieces[Pt.piece(Pt.P, side=side^1)]
                if pawn_from_sqs:
                    # penalty += (MG_PIECES[Pt.base_type(pt)] / 8) - 20
                    # more penalty if the piece is pinned
                    if p in pins:
                        penalty += (MG_PIECES[Pt.base_type(pt)] / 3) - 20
                        # more penalty if the pawn is supported on the attack square
                        if position.attacks[side ^ 1] & pawn_attack_sq or \
                           position.attacks[side] & pawn_attack_sq == 0:
                            penalty += (MG_PIECES[Pt.base_type(pt)] / 3) - 20
    
    return int(penalty * 1 / 4)

def discoveries_and_pins(position, side, target_piece_type=Pt.K):
    """Return squares of singular pieces between sliders and the king of side
    `side`. Blockers of opposite side can move and cause discovered check, and
    blockers of same side are pinned. `target_piece_type` is piece things are pinned to."""
    pinned = []
    discoverers = []
    target_piece_sq = position.pieces[Pt.piece(target_piece_type, side)]
    us = position.occupied[side]
    them = position.occupied[side ^ 1]
    all_occupied = us | them

    diags = PSEUDO_ATTACKS[Pt.B][bit_position(target_piece_sq)]
    diag_sliders = position.pieces[Pt.piece(Pt.B, side ^ 1)]
    diag_sliders |= position.pieces[Pt.piece(Pt.Q, side ^ 1)]
    snipers = diags & diag_sliders

    cross = PSEUDO_ATTACKS[Pt.R][bit_position(target_piece_sq)]
    cross_sliders = position.pieces[Pt.piece(Pt.R, side ^ 1)]
    cross_sliders |= position.pieces[Pt.piece(Pt.Q, side ^ 1)]
    snipers |= cross & cross_sliders
    
    for sniper_sq in iterate_pieces(snipers):
        squares_between = BETWEEN_SQS[bit_position(sniper_sq)][bit_position(target_piece_sq)]
        if count_bits(squares_between & all_occupied) == 1:
            p = us & squares_between
            d = them & squares_between
            if p:
                pinned.append(p)
            else:
                if not position.pieces[Pt.piece(Pt.P, side ^ 1)] & d:
                    discoverers.append(d)

    return (discoverers, pinned)

def pawn_potential_penalty(position, side, potentials):
    potential = potentials[side]
    potential ^= FULL_BOARD
    potential &= (RANKS[2] | RANKS[3] | RANKS[4] | RANKS[5])
    return count_bits(potential) * 20
    
def pawn_attack_potential(p, side):
    """Returns the squares on either side of pawn, ahead of the pawn. Those
    squares can be attacked potentially."""
    p_file, p_rank = get_file(p), get_rank(p)
    left_file = max(p_file - 1, 0)
    right_file = min(p_file + 1, 7)
    above = (p - 1)
    if side == Side.WHITE:
        above = above ^ FULL_BOARD
    above &= (RANKS[p_rank] ^ FULL_BOARD)
    above &= (FILES[left_file] | FILES[right_file])
    return above

def all_pawn_attack_potentials(position, side):
    """Return attack potentials of pawns of side `side`"""
    pawns = position.pieces[Pt.piece(Pt.P, side)]
    potential = 0
    for pawn in iterate_pieces(pawns):
        potential |= pawn_attack_potential(pawn, side)
    return potential
    
def center_attacks_bonus(position, side):
    bonus = 0
    for pt in Pt.piece_types(side=side):
        value = count_bits(position.piece_attacks[pt] & (E4 | E5 | D4 | D5))
        if Pt.base_type(pt) == Pt.P:
            value *= 2
        bonus += value
    return bonus * 10

def evaluate(position, debug=False):
    if ' '.join(map(str, position.moves)) == "e2-e4 e7-e6 Qd1-f3":
        debug = True
    
    evals = defaultdict(lambda: [0, 0])
        
    # Check for mate
    if position.is_mate():
        return -1000000
    
    # TODO: implement stalemate
    
    evaluations = [0, 0]

    counts = piece_counts(position)

    potentials = [all_pawn_attack_potentials(position, Side.WHITE),
                 all_pawn_attack_potentials(position, Side.BLACK)]
    
    for side in [Side.WHITE, Side.BLACK]:

        side_str = "WHITE" if side == Side.WHITE else "BLACK"
        
        # count material
        for base_type in [Pt.P, Pt.N, Pt.B,
                           Pt.R, Pt.Q, Pt.K]:
            piece_type = Pt.piece(base_type, side)
            
            if base_type is not Pt.K:
                value = counts[piece_type] * material_eval(counts, base_type, side)
                if debug: evals["Material %s" % (HUMAN_PIECE[piece_type])][side] += value
                evaluations[side] += value
                
            # Positional bonuses and penalties:

            # ..rook considerations
            if base_type == Pt.R:
                for rook in iterate_pieces(position.pieces[piece_type]):
                    value = rook_position_bonus(rook, position, side)
                    if debug: evals["Rook Position %s" % (HUMAN_PIECE[piece_type])][side] += value
                    evaluations[side] += value
                
            # ..minor outpost, minor behind pawn
            if base_type in [Pt.B, Pt.N]:
                for minor in iterate_pieces(position.pieces[piece_type]):
                    value = minor_outpost_bonus(base_type, position, side, potentials)
                    if debug: evals["Minor Outpost %s" % (HUMAN_PIECE[piece_type])][side] += value
                    evaluations[side] += value

                    if base_type == Pt.B:
                        value = bad_bishop_penalty(minor, position, side)
                        if debug: evals["Bad Bishop Penalty %s" % (HUMAN_PIECE[piece_type])][side] += value
                        evaluations[side] -= value
                
                value = minor_behind_pawn(piece_type, position, side)
                if debug: evals["Minor Behind Pawn %s" % (HUMAN_PIECE[piece_type])][side] += value
                evaluations[side] += value
            
            # ..pawn structure
            if base_type == Pt.P:
                value = pawn_structure(position, side)
                if debug: evals["Pawn Structure"][side] += value
                evaluations[side] += value

                value = pawn_potential_penalty(position, side, potentials)
                if debug: evals["Pawn Potential Penalty"][side] += value
                evaluations[side] -= value
                
            # ..piece-square table adjustments
            if base_type in [Pt.P, Pt.N, Pt.B, Pt.K]:
                value = psqt_value(piece_type, position, side)
                if debug: evals["PSQT adjustments"][side] += value
                evaluations[side] += value

        # center attacks bonus
        value = center_attacks_bonus(position, side)
        if debug: evals["Center Attack Bonus"][side] += value
        evaluations[side] += value
                
        # weak/hanging pieces penalties
        # for ep in next_en_prise(position, side):
        #     pt, *rest = ep
        #     bt = Pt.base_type(pt)
        #     value = (MG_PIECES[bt] / MG_PIECES[Pt.P]) * 30
        #     if debug: evals["En-prise penalties %s" % (HUMAN_PIECE[bt])][side] += value
        #     evaluations[side] -= value
        
        # pins and discoveries to king
        discoverers, pinned = discoveries_and_pins(position, side)
        # pins and discoveries to queen
        q_discoverers, q_pinned = discoveries_and_pins(position, side, Pt.Q)
        
        # unprotected
        # value = unprotected_penalty(position, side, pinned + q_pinned)
        # if debug: evals["Weak/Hanging penalties"][side] += value
        # evaluations[side] -= value

        # threats
        value = threats(position, side, pinned + q_pinned)
        if debug: evals["Threats bonus"][side] += value
        evaluations[side] += value
        
        value = len(pinned) * 15
        if debug: evals["Pins to King penalty"][side] += value
        evaluations[side] -= value
        
        value = len(discoverers) * 150
        if debug: evals["Discovery threats to King penalty"][side] += value
        evaluations[side] -= value
        
        value = len(q_pinned) * 10
        if debug: evals["Pins to Queen penalty"][side] += value
        evaluations[side] -= value
        
        value = 0
        for discoverer in q_discoverers:
            # defended discoverers
            if discoverer & position.attacks[side ^ 1]:
                value += 50
        if debug: evals["Discovery threats to Queen penalty"][side] += value
        evaluations[side] -= value

        # mobility, taking pins to king into account
        value = mobility(position, side, pinned)
        if debug: evals["Mobility"][side] += value
        evaluations[side] += value
        
        # king safety, castle readiness
        value = 0
        if side == Side.WHITE:
            if white_can_castle_kingside(position.position_flags, position.attacks[Side.BLACK], position.occupied[Side.WHITE]):
                value += (2 - count_bits(position.occupied[Side.WHITE] & (F1 | G1))) * 10
            elif white_can_castle_queenside(position.position_flags, position.attacks[Side.BLACK], position.occupied[Side.WHITE]):
                value += (3 - count_bits(position.occupied[Side.WHITE] & (D1 | C1 | B1))) * 10
        else:
            if black_can_castle_kingside(position.position_flags, position.attacks[Side.WHITE], position.occupied[Side.BLACK] ^ FULL_BOARD):
                value += (2 - count_bits(position.occupied[Side.BLACK] & (F8 | G8))) * 10
            elif black_can_castle_queenside(position.position_flags, position.attacks[Side.WHITE], position.occupied[Side.BLACK] ^ FULL_BOARD):
                value += (3 - count_bits(position.occupied[Side.BLACK] & (D8 | C8 | B8))) * 10
        if debug: evals["Castling readiness"][side] += value
        evaluations[side] += value

        king_zone_us = king_safety_squares(position, side)
        king_zone_them = king_safety_squares(position, side ^ 1)

        # .. pawn cover of own king
        value = pawn_cover_bonus(king_zone_us, position, side)
        if debug: evals["Pawn cover"][side] += value
        evaluations[side] += value
        
        # .. king attack bonuses
        value = king_zone_attack_bonus(king_zone_them, position, side)
        if debug: evals["King Attack"][side] += value
        evaluations[side] += value

    if debug: pprint.pprint(dict(evals))
    # pprint.pprint(evals.items())
        
    res_value = int(evaluations[Side.WHITE] - evaluations[Side.BLACK])
    if debug: print("EVAL", res_value)
    if position.white_to_move():
        return res_value
    else:
        return -res_value
