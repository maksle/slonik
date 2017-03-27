import pprint
from collections import defaultdict
from material import MG_PIECES, piece_counts, material_eval
from psqt import psqt_value
from move_gen import *
from piece_type import PieceType as Pt
from position import Position
from bb import *

class Evaluation():
    def __init__(self, position):
        self.position = position
        self.piece_attacks = [0] * 13
        self.all_attacks = [0] * 2
        self.double_attacks = [0] * 2

    def init_attacks(self):
        occ = self.position.occupied[Side.WHITE] | self.position.occupied[Side.BLACK]
        for side in [Side.WHITE, Side.BLACK]:
            pinned = self.position.pinned[Pt.piece(Pt.K, side)]
            king_us = self.position.pieces[Pt.piece(Pt.K, side)]
            for sq in range(64):
                pt = self.position.squares[sq]
                if pt == Pt.NULL: continue
                if Pt.get_side(pt) == side:
                    sq_bb = 1 << sq
                    attacks = piece_attack(pt, sq_bb, occ)
                    if pinned & sq_bb:
                        attacks &= LINE_SQS[bit_position(king_us)][sq]
                    self.piece_attacks[pt] |= attacks
                    self.double_attacks[side] |= self.all_attacks[side] & attacks
                    self.all_attacks[side] |= attacks
        return self
    
    def king_safety_squares(self, side):
        """squares surrounding king plus one rank further"""
        position = self.position
        king_sq = position.pieces[Pt.piece(Pt.K, side)]
        attacks = king_attack(king_sq)
        if side == Side.WHITE:
            attacks |= attacks << 8
        else:
            attacks |= attacks >> 8
        return attacks
    
    def king_zone_attack_bonus(self, king_zones, side):
        """Attackers of enemy king zone, each weighted by piece weight, including
        xray attacks."""
        position = self.position
        us = side
        them = side ^ 1
        their_king = position.pieces[Pt.piece(Pt.K, them)]
        bonus = 0

        if them == Side.WHITE:
            camp = (RANKS[0] | RANKS[1] | RANKS[2] | RANKS[3] | RANKS[4])
        else:
            camp = (RANKS[3] | RANKS[4] | RANKS[5] | RANKS[6] | RANKS[7])

        king_file = get_file(their_king)
        if 0 <= king_file <= 2: files = range(0, 4)
        elif 3 <= king_file <= 4: files = range(2, 6)
        else: files = range(4, 8)
        flank = 0
        for f in files: flank |= FILES[f]
        flank &= camp

        # defended by their king only
        king_defended = self.piece_attacks[Pt.piece(Pt.K, them)] & invert(self.double_attacks[them])
        bonus += count_bits(king_defended) * 8

        # attacks directly around their king
        bonus += count_bits(self.all_attacks[us] & king_attack(their_king)) * 4
        # bonus for the more coordinated attacks in the king zone
        bonus += count_bits(self.double_attacks[us] & king_zones[them]) * 4

        # king flank huddling bonus
        bonus += count_bits(self.all_attacks[us] & flank) * 4
        # extra bonus for double attacks in flank, not defended by pawn
        bonus += count_bits(self.all_attacks[us] & flank & self.double_attacks[us] & invert(self.piece_attacks[Pt.piece(Pt.P, them)])) * 7

        # safe positions to check from
        safe = invert(self.all_attacks[them] | position.occupied[us])

        # safe positions to check from b/c protected by their queen only 
        safe2 = safe | self.piece_attacks[Pt.piece(Pt.Q, them)] & invert(self.double_attacks[them]) & self.double_attacks[us]

        bishop_rays = PSEUDO_ATTACKS[Pt.B][bit_position(their_king)]
        rook_rays = PSEUDO_ATTACKS[Pt.R][bit_position(their_king)]

        # potential safe knight check squares
        safe_n_checks = knight_attack(their_king) & safe2 & self.piece_attacks[Pt.piece(Pt.N, us)]
        if safe_n_checks: bonus += 25

        # potential safe bishop check squares
        safe_b_checks = bishop_rays & safe2 & self.piece_attacks[Pt.piece(Pt.B, them)]
        if safe_b_checks: bonus += 15

        # potential safe rook check squares
        safe_r_checks = rook_rays & safe2 & self.piece_attacks[Pt.piece(Pt.R, them)]
        if safe_b_checks: bonus += 20

        # potential safe queen check squares
        safe_q_checks = (bishop_rays | rook_rays) & safe & self.piece_attacks[Pt.piece(Pt.Q, them)]
        if safe_b_checks: bonus += 20

        # safe queen contact checks
        safe_q_contact_checks = king_defended & self.double_attacks[us] & self.piece_attacks[Pt.piece(Pt.Q, us)]
        if safe_q_contact_checks: bonus += 35

        # attack more difficult without the queen
        if position.pieces[Pt.piece(Pt.Q, us)] == 0:
            bonus /= 2

        return int(bonus)

    def pawn_cover_bonus(self, king_zones, side):
        position = self.position
        pawn_type = Pt.piece(Pt.P, side)
        pawn_cover = king_zones[side] & position.pieces[pawn_type]
        # increase this too much, king will move to get close to e4,d4 pawns! or only do this when king is castled?
        return count_bits(pawn_cover) * 6

    def rook_position_bonus(self, rook, side):
        position = self.position
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
        if self.piece_attacks[Pt.piece(Pt.R, side)] & rook:
            bonus += 15

        return bonus

    def pawns_evaluation(self, side):
        position = self.position
        us, them = side, side ^ 1
        pawns_us = position.pieces[Pt.piece(Pt.P, us)]
        pawns_them = position.pieces[Pt.piece(Pt.P, them)]

        score = 0

        files = set(get_file(p) for p in iterate_pieces(pawns_us))

        protected = (shift_ne(pawns_us, us) | shift_nw(pawns_us, us)) & pawns_us
        phalanx = (shift_east(pawns_us, us) | shift_west(pawns_us, us)) & pawns_us
        connected = protected | phalanx

        isolated = 0
        opposed = 0
        for f in files:
            fl = 1 if f == 0 else f - 1
            fr = 6 if f == 7 else f + 1
            if not (FILES[fl] & pawns_us) | (FILES[fr] & pawns_us):
                isolated |= FILES[f] & pawns_us
            if FILES[f] & pawns_them:
                opposed |= FILES[f] & pawns_us

        doubled = 0
        for f in files:
            doubled |= reset_ls1b(pawns_us & FILES[f])

        score -= count_bits(isolated & opposed) * 30
        score -= count_bits(isolated ^ opposed) * 40
        for c in iterate_pieces(connected):
            val = 2 ** get_rank(c, us)
            if c & phalanx:
                val += val - (2 ** (get_rank(c, us) - 1))
            if c & opposed:
                val >> 1
            score += val
        score -= count_bits(doubled) * 25
        # TODO: add backward pawns
        # don't recount these if backward
        score -= count_bits(pawns_us ^ protected) * 15
        return score

    def minor_outpost_bonus(self, minor, side, potentials):
        position = self.position
        if minor == 0: return 0
        us = side
        them = side ^ 1
        potential_them = potentials[them]
        potential_us = potentials[us]
        base_type = Pt.base_type(minor)
        pawns_us = position.pieces[Pt.piece(Pt.P, side)]
        outpost_ranks = [RANKS[3] | RANKS[4] | RANKS[5],
                         RANKS[2] | RANKS[3] | RANKS[4]]
        outpost_squares = outpost_ranks[side] & invert(potential_them) & potential_us
        if outpost_squares:
            # reachable squares
            if base_type == Pt.N: bonus = 12
            else: bonus = 7

            # successfully outposted
            if minor & outpost_squares: bonus += 35

            return bonus
        return 0

    def bad_bishop_penalty(self, minor, side):
        position = self.position
        color = minor & DARK_SQUARES
        pawns_us = position.pieces[Pt.piece(Pt.P, side)]
        if color == 0: pawns = pawns_us & WHITE_SQUARES
        else: pawns = pawns_us & DARK_SQUARES
        return count_bits(pawns) * 8

    def minor_behind_pawn(self, minors, side):
        position = self.position
        pawns_in_front = position.pieces[Pt.piece(Pt.P, side)] & shift_north(minors, side)
        pawns_in_front &= (RANKS[2] | RANKS[3] | RANKS[4] | RANKS[5])
        return count_bits(pawns_in_front) * 8

    def mobility(self, side, PINNED):
        """Bonus for legal moves not attacked by lower weight piece. Pinned pieces
        have restricted mobility"""
        position = self.position
        mobility = 0
        piece_types = [Pt.P, Pt.N, Pt.B, Pt.R, Pt.Q]

        pinned_piece_types = []
        if PINNED[Pt.piece(Pt.K, side)]:
            pinned_piece_types = [position.squares[bit_position(p)] for p in iterate_pieces(PINNED[Pt.piece(Pt.K, side)])]

        for base_pt in piece_types:
            pt = Pt.piece(base_pt, side)

            # if attacked by lower weight piece, it doesn't count
            lower_wts = (pt for pt in piece_types if pt < base_pt)
            opp_attacks = 0
            for piece_type in lower_wts:
                opp_pt = Pt.piece(piece_type, side ^ 1)
                opp_attacks |= self.piece_attacks[opp_pt]
            attacks = self.piece_attacks[pt]
            attacks &= invert(opp_attacks)
            attacks &= invert(position.occupied[side])
            mobility_factor = base_pt if base_pt < Pt.K else 1
            mobility += count_bits(attacks) * mobility_factor

        return mobility

    def attacked_pieces(self, side):
        position = self.position
        return position.occupied[side] & self.all_attacks[side ^ 1]

    def threats(self, side, PINNED):
        position = self.position
        occupied = position.occupied[Side.WHITE] | position.occupied[Side.BLACK]
        free = occupied ^ FULL_BOARD
        rank2 = RANKS[1] if side == Side.WHITE else RANKS[6]

        us = side
        them = side ^ 1

        bonus = 0
        penalty = 0
        # them_no_qk = position.occupied[side ^ 1] \
        #     ^ position.pieces[Pt.piece(Pt.Q, side ^ 1)] \
        #     ^ position.pieces[Pt.piece(Pt.K, side ^ 1)]
        # loose_pieces = them_no_qk & ((self.all_attacks[side] | self.all_attacks[side]) ^ FULL_BOARD)
        # if loose_pieces:
        #     bonus += 25

        # The following is copied as in stockfish:

        # non-pawn enemies attacked by pawn
        weak = (position.occupied[them] ^ position.pieces[Pt.piece(Pt.P, them)]) \
               & (self.piece_attacks[Pt.piece(Pt.P, us)])

        if weak:
            # our pawns protected by us or not attacked by them
            b = position.pieces[Pt.piece(Pt.P, us)] & (self.all_attacks[us] | invert(self.all_attacks[side]))

            safe_threats = (shift_ne(b, us) | shift_nw(b, us)) & weak

            if weak ^ safe_threats:
                bonus += 70

            for threatened_piece in iterate_pieces(safe_threats):
                bonus += 150
                if Pt.base_type(position.squares[bit_position(threatened_piece)]) in [Pt.R, Pt.Q]:
                    bonus += 50

        # non-pawn enemies defended by pawn
        defended = (position.occupied[them] ^ position.pieces[Pt.piece(Pt.P, them)]) \
               & (self.piece_attacks[Pt.piece(Pt.P, them)])

        # enemies not defended by a pawn and under our attack
        weak = position.occupied[them] \
               & invert(self.piece_attacks[Pt.piece(Pt.P, them)]) \
               & self.all_attacks[us]

        if defended | weak:
            # minor attacks
            minor_attack = self.piece_attacks[Pt.piece(Pt.N, us)] | self.piece_attacks[Pt.piece(Pt.B, us)]
            b = (defended | weak) & minor_attack
            for attacked in iterate_pieces(b):
                attacked_type = Pt.base_type(position.squares[bit_position(attacked)])
                if attacked_type == Pt.N:
                    bonus += 10
                if attacked_type > Pt.N:
                    bonus += 56
                if attacked_type == Pt.Q:
                    bonus += 40

            # rook attacks
            b = (position.pieces[Pt.piece(Pt.Q, them)] | weak) & self.piece_attacks[Pt.piece(Pt.R, us)]
            for attacked in iterate_pieces(b):
                attacked_type = Pt.base_type(position.squares[bit_position(attacked)])
                if attacked_type > Pt.P and attacked_type != Pt.R:
                    bonus += 40

            # hanging
            bonus += 44 * count_bits(weak & invert(self.all_attacks[them]))

            # king attacks
            b = weak & self.piece_attacks[Pt.piece(Pt.K, us)]
            more_than_one = reset_ls1b(b) > 0
            if more_than_one: bonus += 18 # 120 for endgame
            elif b: bonus += 6 # 60 for endgame

        # bonus for pawn push that attacks pieces
        # pawns already attacking were considered earlier above
        b = position.pieces[Pt.piece(Pt.P, us)]
        b = shift_north(b | (shift_north(b & rank2, us) & free), us)
        b &= free & (self.all_attacks[us] | invert(self.all_attacks[them]))
        b = (shift_ne(b, us) | shift_nw(b, us)) & invert(self.piece_attacks[Pt.piece(Pt.P, us)])
        b2 = PINNED[Pt.piece(Pt.K, side ^ 1)] | PINNED[Pt.piece(Pt.Q, side ^ 1)]
        bonus += count_bits(b & b2 & position.occupied[them]) * 70 + count_bits(b & invert(b2) & position.occupied[them]) * 20

        return bonus

    def unprotected_penalty(self, side, pins):
        position = self.position
        us = position.occupied[side]
        them = position.occupied[side ^ 1]
        free = (us | them) ^ FULL_BOARD
        us_attacked = attacked_pieces(position, side)
        penalty = 0
        for pt in Pt.piece_types(side=side):
            num = count_bits(position.pieces[pt] & us_attacked)
            penalty += num * 10

            defended = us_attacked & position.pieces[pt] & self.all_attacks[side]
            for defended_piece in iterate_pieces(defended):
                if defended_piece & self.piece_attacks[Pt.piece(Pt.P, side=side)]:
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
                            if self.all_attacks[side ^ 1] & pawn_attack_sq or \
                               self.all_attacks[side] & pawn_attack_sq == 0:
                                penalty += (MG_PIECES[Pt.base_type(pt)] / 3) - 20

        return int(penalty * 1 / 4)

    def pawn_potential_penalty(self, side, potentials):
        position = self.position
        potential = potentials[side]
        potential ^= FULL_BOARD
        potential &= (RANKS[2] | RANKS[3] | RANKS[4] | RANKS[5])
        return count_bits(potential) * 2

    def pawn_attack_potential(self, p, side):
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

    def all_pawn_attack_potentials(self, side):
        """Return attack potentials of pawns of side `side`"""
        position = self.position
        pawns = position.pieces[Pt.piece(Pt.P, side)]
        potential = 0
        for pawn in iterate_pieces(pawns):
            potential |= self.pawn_attack_potential(pawn, side)
        return potential

    def center_attacks_bonus(self, side):
        position = self.position
        bonus = 0
        for pt in Pt.piece_types(side=side):
            value = count_bits(self.piece_attacks[pt] & (E4 | E5 | D4 | D5))
            if Pt.base_type(pt) == Pt.P:
                value *= 2
            bonus += value
        return bonus * 5
    
    def evaluate(self, debug=False):
        position = self.position
        # if ' '.join(map(str, position.moves)) == "e2-e4 e7-e6 Qd1-f3":
        #     debug = True

        evals = defaultdict(lambda: [0, 0])

        # Check for mate
        # if position.is_mate():
        #     return -1000000

        # TODO: implement stalemate

        evaluations = [0, 0]

        # self.init_attacks()
        
        counts = piece_counts(position)

        POTENTIALS_BB = [self.all_pawn_attack_potentials(Side.WHITE), self.all_pawn_attack_potentials(Side.BLACK)]
        KING_ZONE_BB = [self.king_safety_squares(Side.WHITE), self.king_safety_squares(Side.BLACK)]
        
        q_discoverers, q_pinned, q_sliding_checkers = self.position.get_discoveries_and_pins(Pt.Q)
        PINNED = self.position.pinned
        PINNED[Pt.piece(Pt.Q, Side.WHITE)] = q_pinned[Pt.piece(Pt.Q, Side.WHITE)]
        PINNED[Pt.piece(Pt.Q, Side.BLACK)] = q_pinned[Pt.piece(Pt.Q, Side.BLACK)]
        DISCOVERERS = self.position.discoverers
        
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
                        value = self.rook_position_bonus(rook, side)
                        if debug: evals["Rook Position %s" % (HUMAN_PIECE[piece_type])][side] += value
                        evaluations[side] += value

                # ..minor outpost, minor behind pawn
                if base_type in [Pt.B, Pt.N]:
                    for minor in iterate_pieces(position.pieces[piece_type]):
                        value = self.minor_outpost_bonus(base_type, side, POTENTIALS_BB)
                        if debug: evals["Minor Outpost %s" % (HUMAN_PIECE[piece_type])][side] += value
                        evaluations[side] += value

                        if base_type == Pt.B:
                            value = self.bad_bishop_penalty(minor, side)
                            if debug: evals["Bad Bishop Penalty %s" % (HUMAN_PIECE[piece_type])][side] += value
                            evaluations[side] -= value

                    value = self.minor_behind_pawn(piece_type, side)
                    if debug: evals["Minor Behind Pawn %s" % (HUMAN_PIECE[piece_type])][side] += value
                    evaluations[side] += value

                # ..pawn structure
                if base_type == Pt.P:
                    value = self.pawns_evaluation(side)
                    if debug: evals["Pawn Structure"][side] += value
                    evaluations[side] += value

                    value = self.pawn_potential_penalty(side, POTENTIALS_BB)
                    if debug: evals["Pawn Potential Penalty"][side] += value
                    evaluations[side] -= value

                # ..piece-square table adjustments
                if base_type in [Pt.P, Pt.N, Pt.B, Pt.K]:
                    value = psqt_value(piece_type, position, side)
                    if debug: evals["PSQT adjustments"][side] += value
                    evaluations[side] += value

            # center attacks bonus
            value = self.center_attacks_bonus(side)
            if debug: evals["Center Attack Bonus"][side] += value
            evaluations[side] += value

            # weak/hanging pieces penalties
            # for ep in next_en_prise(side):
            #     pt, *rest = ep
            #     bt = Pt.base_type(pt)
            #     value = (MG_PIECES[bt] / MG_PIECES[Pt.P]) * 30
            #     if debug: evals["En-prise penalties %s" % (HUMAN_PIECE[bt])][side] += value
            #     evaluations[side] -= value

            # unprotected
            # value = unprotected_penalty(side, pinned + q_pinned)
            # if debug: evals["Weak/Hanging penalties"][side] += value
            # evaluations[side] -= value

            # threats
            value = self.threats(side, PINNED)
            if debug: evals["Threats bonus"][side] += value
            evaluations[side] += value

            value = count_bits(PINNED[Pt.piece(Pt.K, side)]) * 15
            if debug: evals["Pins to King penalty"][side] += value
            evaluations[side] -= value

            value = count_bits(DISCOVERERS[Pt.piece(Pt.K, side ^ 1)]) * 150
            if debug: evals["Discovery threats to King bonus"][side] += value
            evaluations[side] += value

            value = count_bits(q_pinned[Pt.piece(Pt.Q, side)]) * 10
            if debug: evals["Pins to Queen penalty"][side] += value
            evaluations[side] -= value

            value = count_bits(q_discoverers[Pt.piece(Pt.Q, side ^ 1)]) * 100
            if debug: evals["Discovery threats to Queen bonus"][side] += value
            evaluations[side] += value

            # mobility, taking pins to king into account
            value = self.mobility(side, PINNED)
            if debug: evals["Mobility"][side] += value
            evaluations[side] += value

            # king safety, castle readiness
            value = 0
            if side == Side.WHITE:
                if white_can_castle_kingside(position.position_flags, self.all_attacks[Side.BLACK], position.occupied[Side.WHITE]):
                    value += (2 - count_bits(position.occupied[Side.WHITE] & (F1 | G1))) * 4
                elif white_can_castle_queenside(position.position_flags, self.all_attacks[Side.BLACK], position.occupied[Side.WHITE]):
                    value += (3 - count_bits(position.occupied[Side.WHITE] & (D1 | C1 | B1))) * 4
            else:
                if black_can_castle_kingside(position.position_flags, self.all_attacks[Side.WHITE], position.occupied[Side.BLACK] ^ FULL_BOARD):
                    value += (2 - count_bits(position.occupied[Side.BLACK] & (F8 | G8))) * 4
                elif black_can_castle_queenside(position.position_flags, self.all_attacks[Side.WHITE], position.occupied[Side.BLACK] ^ FULL_BOARD):
                    value += (3 - count_bits(position.occupied[Side.BLACK] & (D8 | C8 | B8))) * 4
            if debug: evals["Castling readiness"][side] += value
            evaluations[side] += value

            # .. pawn cover of own king
            value = self.pawn_cover_bonus(KING_ZONE_BB, side)
            if debug: evals["Pawn cover"][side] += value
            evaluations[side] += value

            # .. king attack bonuses
            value = self.king_zone_attack_bonus(KING_ZONE_BB, side)
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

        captured_piece_type = position.squares[bit_position(square)]
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

def lowest_attacker(pos, square, side=None, highest_attacker=Pt.NULL):
    """Finds lowest piece of `side` attacking a square. `highest_attacker` means
    skip attacks of higher or equal value of given pt"""
    position = Position(pos)
    side = position.side_to_move() if side is None else side
    highest_attacker = Pt.K if highest_attacker == Pt.NULL else highest_attacker

    # pawn
    if Pt.P < highest_attacker:
        possible_from_squares = shift_sw(square, side) | shift_se(square, side)
        piece_type = Pt.piece(Pt.P, side)
        attackers = position.pieces[piece_type] & possible_from_squares
        if attackers:
            return piece_type, ls1b(attackers)

    # knight
    if Pt.N < highest_attacker:
        piece_type = Pt.piece(Pt.N, side)
        possible_from_squares = knight_attack(square)
        attackers = position.pieces[piece_type] & possible_from_squares
        if attackers:
            return piece_type, ls1b(attackers)

    # sliders
    for piece_type in [Pt.B, Pt.R]:
        if piece_type < highest_attacker:
            pt = Pt.piece(piece_type, side)
            occ = position.occupied[side] | position.occupied[side^1]
            attacks = piece_attack(pt, square, occ)

            b = attacks & position.pieces[pt]
            if b: return pt, ls1b(b)

            if Pt.Q < highest_attacker:
                qn = Pt.piece(Pt.Q, side)
                b = attacks & position.pieces[qn]
                if b: return qn, ls1b(b)

    # king
    if Pt.K < highest_attacker:
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
            # equal val but undefended more readily handled by eval_see
            lowest = lowest_attacker(pos, square, side ^ 1, Pt.base_type(pt))
            if not lowest: continue
            attacker_pt, attacker_sq = lowest
            return pt, square, attacker_pt, attacker_sq
    return 0, 0, 0, 0
