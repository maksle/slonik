from IPython import embed
from collections import namedtuple
from position import *
from constants import *
from collections import defaultdict
from material import *
from psqt import *
# from evals import eval_see, rook_position_bonus, pawn_cover_bonus, king_safety_squares, king_zone_attack_bonus
from evals import *
from operator import itemgetter
import itertools
import tt

class SearchInfo():
    def __init__(self):
        self.ply = 0
        self.current_variation = []
        self.pv = []
        self.null_move_prune_search = False

class SearchPos():
    def __init__(self, position):
        self.position = Position(position)
    def is_leaf(self):
        return self.position.is_mate()
    def value(self):
        return evaluate(self.position)
    def children(self, si, ply, quiescence=False):
        moves = self.position.generate_moves()
        smoves = sort_moves(moves, self.position, si, ply, quiescence)
        yield from smoves
    def last_move(self):
        return self.position.last_move()

def evaluate_moves(position):
    moves = list(position.generate_moves())
    scores = []
    for move in moves:
        pos = Position(position)
        pos.make_move(move)
        scores.append(-evaluate(pos))
    res = list(sorted(zip(map(str,moves), scores), key=lambda x: x[1]))
    import pprint
    return pprint.pprint(res)

# history moves
History = namedtuple('History', ['move', 'value'])
move_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_history(side, move, value):
    entry = move_history[side][move.piece_type][bit_position(move.to_sq)]
    adjusted_val = value
    if entry:
        adjusted_val = entry.value + .2 * (value - entry.value)
    move_history[side][move.piece_type][bit_position(move.to_sq)] = History(move, adjusted_val)
def lookup_history(side, move):
    if move and move.piece_type is not PieceType.NULL.value:
        return move_history[side][move.piece_type][bit_position(move.to_sq)]
    
# counter moves
Counter = namedtuple('Counter', ['move'])
counter_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_counter(side, move, counter):
    entry = counter_history[side][move.piece_type][bit_position(move.to_sq)]
    # adjusted_val = int(value * 100)
    # if entry:
    #     adjusted_val = int(entry.value * .7 + value * 100 * .3)
    counter_history[side][move.piece_type][bit_position(move.to_sq)] = Counter(counter)
def lookup_counter(side, move):
    if move and move.piece_type is not PieceType.NULL.value:
        return counter_history[side][move.piece_type][bit_position(move.to_sq)]
    
tb_hits = 0
node_count = 0
killer_moves = defaultdict(list)

def add_killer(ply, move):
    if move in killer_moves[ply]:
        return
    if len(killer_moves[ply]) == 3:
        killer_moves[ply].pop()
    killer_moves[ply].append(move)

def iterative_deepening(thres_confidence, node):
    alpha = LOW_BOUND
    beta = HIGH_BOUND
    
    target_thres_p = (1 - thres_confidence)
    curr_thres_p = 1
    
    exp = 0
    val = 0
    si = [None] * 64
    
    while curr_thres_p > target_thres_p:
        exp += 1
        curr_thres_p /= 2
        finished = False
        fail_factor = 18

        alpha, beta = val - fail_factor, val + fail_factor
        
        # print (">> depth", depth)
        print (">> curr_thres_p", curr_thres_p, "target_thres_p", target_thres_p)
        
        while not finished:
            val = search(node, si, 0, alpha, beta, 1, curr_thres_p, True)

            if val < alpha:
                print("faillow", "a", alpha, "b", beta, "val", val, "factor", fail_factor)
            if val > beta:
                print("failhigh", "a", alpha, "b", beta, "val", val, "factor", fail_factor)

            if val <= alpha:
                alpha = val - fail_factor
                beta = (alpha + beta) // 2
                fail_factor += fail_factor // 3 + 6
            elif val >= beta:
                alpha = (alpha + beta) / 2
                beta = val + fail_factor
                fail_factor += fail_factor // 3 + 6
            else:
                finished = True
    
    global node_count
    print("node count", node_count)
    print("pv:", end=" ")
    print_moves(si[0].pv)
    print_moves(find_pv(node.position))
    return val
    
def search(node, si, ply, a, b, curr_p, thres_p, pv_node):
    global node_count
    node_count += 1
    
    assert(pv_node or a == b-1)
    
    a_orig = a
    is_root = pv_node and ply == 0
    
    si[ply] = SearchInfo()
    si[ply+1] = SearchInfo()
    
    found = False
    if not pv_node:
        found, tt_ind, tt_entry = tt.get_tt_index(node.position.zobrist_hash)
        if found and tt_entry.prob >= curr_p:
            if (tt_entry.bound_type == tt.BoundType.LO_BOUND.value and tt_entry.value >= b) \
               or (tt_entry.bound_type == tt.BoundType.HI_BOUND.value and tt_entry.value < a) \
               or tt_entry.bound_type == tt.BoundType.EXACT.value:
                move = Move.move_uncompacted(tt_entry.move)
                return tt_entry.value
    
    if curr_p <= thres_p:
        score = qsearch(node, si, ply, a, b, curr_p, thres_p, pv_node, True)
        return score
    
    # if not in_check:
    #     # null move pruning.. if pass move and we're still failing high, dont bother searching
    #     if not pv_node and not si.null_move_prune_search and node.value() >= b:
    #         if depth <= 2: nmp_depth = depth
    #         else: nmp_depth = max(depth-2, 2)

    #         si.null_move_prune_search = True
    #         si.ply += 1
    #         node.position.toggle_side_to_move()
    #         val, chosen_move = search(node, si, -b, -b+1, nmp_depth, False)
    #         val = -val
    #         node.position.toggle_side_to_move()
    #         si.ply -=1
    #         si.null_move_prune_search = False

    #         if val >= b:
    #             return val
        
    best_move = None
    best_move_is_capture = False
    best_val = LOW_BOUND
    move_count = 0

    # if str(node.last_move()) == "Nc6-d4":
    #     print("debug")
    
    moves = list(node.children(si, ply))
    for move in moves:
        # if ' '.join(map(str,node.position.moves)) == "" and str(move) == "Qg4-g2+":
        if str(move) == "Bc8-e6":
            print("debug")
        
        child = SearchPos(move.position)
        
        move_count += 1
        extensions = 0

        is_capture = move.to_sq & node.position.occupied[child.position.side_to_move()]
        see_score = move.see_score or eval_see(node.position, move)
        gives_check = child.position.in_check()
        
        # prune moves with too-negative SEE; more lenient with moves closer to root
        # if see_score + 250 < -50 * lmr_depth * lmr_depth:
        #     # print("pruning", see_score, lmr_depth, list(map(str,node.position.moves)), move)
        #     continue
        
        # TODO?: if non rootNode and we're losing, only look at checks/big captures >= alpha 
        
        # if lmr_depth < 3 and not is_capture:
        #     counter = lookup_counter(node.position.side_to_move(), move)
        #     history = lookup_history(node.position.side_to_move(), move)
        #     # TODO: check that counter is legal?
        #     if counter and history and history.value < -10:
        #         print("hist val", history.value, "cnter", counter.move, "pruning", move)
        #         continue
        #         # lmr_depth = max(lmr_depth - 1, 0)

        # Probabilistic version of LMR
        # .. zero window search reduced 
        if curr_p > .2 and not is_capture:
            val = -search(child, si, ply+1, -(a+1), -a, curr_p * move.prob * .5, thres_p, False)
            do_full_zw = val > a
        else:
            do_full_zw = not (pv_node and (move.prob >= .4))
        
        # .. zero window full allotment search
        if do_full_zw:
            val = -search(child, si, ply+1, -(a+1), -a, curr_p * move.prob, thres_p, True)
            
        # .. full window full allotment search
        # otherwise we let the fail highs cause parent to fail low and try different move
        if pv_node and (move.prob >= .4 or a < val < b):
            val = -search(child, si, ply+1, -b, -a, curr_p * move.prob, thres_p, True)
            
        if val > best_val:
            best_val, best_move = val, move
            # if move_count > 5:
            #     print("movecount", move_count, move, list(map(str,node.position.moves)))
            best_move_is_capture = is_capture
            if val > a and pv_node:
                si[ply].pv = [move] + si[ply+1].pv
                # si.pv = si.current_variation
                # si.pv = si.pv[:si.ply] + si.current_variation[si.ply:]
                if is_root:
                    print_moves(si[ply].pv)
                
        a = max(a, val)
        if a >= b:
            if best_move: 
                add_killer(ply, best_move)
            break

    prior_move = node.position.last_move()
    if len(moves) == 0:
        # mate or statemate
        best_val = node.value()
    elif best_move:
        if not best_move_is_capture:
            bonus = (curr_p * 100) * (curr_p * 100)
            update_history(node.position.side_to_move(), best_move, bonus)
            if prior_move and prior_move.piece_type != PieceType.NULL.value:
                update_counter(node.position.side_to_move() ^ 1, prior_move, best_move)
                # penalize prior move that allowed this good move
                # TODO: make sure that move was a quiet move?
                if len(node.position.moves) > 1 and node.position.moves[-2] != PieceType.NULL.value:
                    update_history(node.position.side_to_move(), node.position.moves[-2], -(bonus + 4))
    elif curr_p > .2 and not best_move_is_capture:
        assert(best_val <= alpha)
        # reward the move that caused this node to fail low
        # TODO: make sure node.moves[-2] was a quiet move?
        # TODO: maybe don't reward at too shallow depth (less confident of it)
        # bonus = (depth * depth) + (2 * depth) - 2
        bonus = (curr_p * 100) * (curr_p * 100)
        if len(node.position.moves) > 1 and node.position.moves[-2] != PieceType.NULL.value:
            update_history(node.position.side_to_move(), node.position.moves[-2], bonus)

    assert(best_val > LOW_BOUND)
    
    best_move = best_move or Move(PieceType.NULL.value)
    if best_val <= a_orig: bound_type = tt.BoundType.HI_BOUND.value
    elif best_val >= b: bound_type = tt.BoundType.LO_BOUND.value
    else: bound_type = tt.BoundType.EXACT.value
    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, best_move.compact(),
                                bound_type, best_val, curr_p))
    
    return best_val

def qsearch(node, si, ply, alpha, beta, curr_p, thres_p, pv_node, entry_node):
    global node_count
    assert(pv_node or alpha == beta-1)
    assert(not entry_node or curr_p <= thres_p)
    node_count += 1

    if pv_node:
        si[ply] = SearchInfo()
        si[ply+1] = SearchInfo()
    
    if entry_node:
        curr_p = 1

    tt_hit = False
    a_orig = alpha
    
    if not pv_node:
        tt_hit, tt_ind, tt_entry = tt.get_tt_index(node.position.zobrist_hash)
        if tt_hit and tt_entry.prob <= curr_p:
            if tt_entry.bound_type == tt.BoundType.EXACT.value:
                return tt_entry.value
            
            if tt_entry.bound_type == tt.BoundType.LO_BOUND.value and tt_entry.value >= beta:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.bound_type == tt.BoundType.HI_BOUND.value and tt_entry.value < alpha:
                beta = min(beta, tt_entry.value)

            if alpha >= beta:
                return tt_entry.value

    in_check = node.position.in_check()

    # to stop search of endless checks, including repetition checks
    if in_check:
        num_moves = len(node.position.moves)
        if curr_p < thres_p and num_moves > 3 \
           and node.position.moves[-3].move_type == MoveType.check.value:
            return node.value()
    
    best_move = Move(PieceType.NULL.value)
    if in_check:
        best_value = start_val = LOW_BOUND
    else:
        best_value = start_val = node.value()

    if best_value >= beta:
        # "stand pat"
        if not tt_hit:
            tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash,
                                        Move(PieceType.NULL.value, None, None).compact(),
                                        tt.BoundType.LO_BOUND.value, best_value, curr_p))
            return best_value
    
    if pv_node and best_value > alpha:
        alpha = best_value
    
    score = None
    move_count = 0
        
    moves = list(node.children(si, ply, quiescence=True))
    for move in moves:
        # if ' '.join(map(str,node.position.moves)) == "Nc6-d4":
        #     print("debug")

        child = SearchPos(move.position)

        move_count += 1
        gives_check = child.position.in_check()
        
        is_capture = move.to_sq & node.position.occupied[node.position.side_to_move() ^ 1]
        
        if not in_check and not gives_check:
            # Futility pruning
            # .. try to avoid calling eval_see
            pt_captured = node.position.squares[bit_position(move.to_sq)]
            if start_val + MG_PIECES[PieceType.base_type(pt_captured)] + MG_PIECES[PieceType.P.value] <= alpha:
                continue
            static_eval = move.see_score if move.see_score is not None else eval_see(node.position, move)
            if start_val + static_eval + MG_PIECES[PieceType.P.value] <= alpha \
               and static_eval < 0:
                continue

        if not in_check or not is_capture:
            static_eval = move.see_score if move.see_score is not None else eval_see(node.position, move)
            if static_eval < 0:
                continue
            
        print("qsearch", end=" ")
        print_moves(child.position.moves)
        score = -qsearch(child, si, ply+1, -beta, -alpha, curr_p * move.prob, thres_p, pv_node, False)
        
        if score > best_value:
            best_value = score
            if score > alpha:
                if pv_node and score < beta:
                    alpha = score
                    best_move = move
                    si[ply].pv = [move] + si[ply+1].pv
                else:
                    assert score >= beta
                    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, move.compact(),
                                                tt.BoundType.LO_BOUND.value, best_value, curr_p))
                    return score
        
    if len(moves) == 0:
        # mate or stalemate
        best_value = node.value()
        # if not (best_value == -1000000 or best_value == 0):
        #     embed()
        assert(not in_check or (best_value == -1000000 or best_value == 0))

    if pv_node and best_value > a_orig: bound_type = tt.BoundType.EXACT.value
    else: bound_type = tt.BoundType.HI_BOUND.value
    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, best_move.compact(),
                                bound_type, best_value, curr_p))
    return best_value

def find_pv(root_pos):
    moves = []
    pos = Position(root_pos)
    
    def find_next_move():
        found, tt_ind, tt_entry = tt.get_tt_index(pos.zobrist_hash)
        if found:
            move = Move.move_uncompacted(tt_entry.move)
            if move != 0:
                return True, move
            else:
                return False, move
        return False, 0

    found, move = find_next_move()
    while found and len(moves) < 20:
        moves.append(move)
        pos.make_move(move)
        found, move = find_next_move()
        
    return moves

def sort_moves(moves, position, si, ply, quiescence):
    from_pv = [] 
    captures = []
    killers = []
    counters = []
    other_moves = []
    checks = []

    side = position.side_to_move()
    other = position.occupied[side ^ 1]
    counter = lookup_counter(side ^ 1, position.last_move())

    ep_default_value = (0, 0, 0, 0)
    ep_before = next(next_en_prise(position, position.side_to_move()), ep_default_value)
    
    def sort_crit(move, en_prise_sort=False):
        entry = lookup_history(side, move)
        see_val = eval_see(position, move)
        if en_prise_sort:
            ep_after_gen = next_en_prise(position, position.side_to_move(), move)
            pt, *rest = ep_before
            pt2, *rest2 = next(ep_after_gen, ep_default_value)
            loss = pt - pt2
            see_val += loss
            
        hist_val = entry.value if entry else 0
        psqt_val = psqt_value_sq(move.piece_type, move.to_sq, position.side_to_move())
        return (see_val, hist_val, psqt_val)
    
    for move in moves:
        if quiescence:
            if is_capture(move.to_sq, other):
                captures.append(move)
            else:
                in_check = position.in_check()
                try_move = move.position
                if in_check and not try_move.in_check(side):
                    # evade check
                    other_moves.append(move)
                elif move.move_type == MoveType.check.value:
                    # give check
                    checks.append(move)
        else:
            if move in si[0].pv + si[ply].pv:
                from_pv.append(move)
            elif is_capture(move.to_sq, other):
                captures.append(move)
            elif move.move_type == MoveType.check.value:
                checks.append(move)
            elif counter and counter.move == move:
                counters.append(move)
            elif move in killer_moves[ply]:
                killers.append(move)
            else:
                other_moves.append(move)
                
    captures = sorted(captures, key=sort_crit, reverse=True)
    checks = sorted(checks, key=sort_crit, reverse=True)
    captures_see = map(lambda c: (sort_crit(c), c), captures)
    sorted_cap_see = sorted(captures_see, key=itemgetter(0), reverse=True)

    other_moves.sort(key=lambda m: sort_crit(m, en_prise_sort=True), reverse=True)
    
    if quiescence:
        result = list(itertools.chain(map(itemgetter(1), sorted_cap_see),
                                      checks,
                                      other_moves))
    else:
        cap_see_gt0 = []
        cap_see_lt0 = []
        cap_see_eq0 = []
        for cs in sorted_cap_see:
            see, hist, psqt = cs[0]
            if see > 0:
                cap_see_gt0.append(cs[1])
            elif see == 0:
                cap_see_eq0.append(cs[1])
            else:
                cap_see_lt0.append(cs[1])

        # counters = [c.move for c in sorted(counters, key=lambda c: c.value, reverse=True)]
        
        result = list(itertools.chain(from_pv, cap_see_gt0, cap_see_eq0, checks, counters, killers, other_moves, cap_see_lt0))

    # guassian formula
    if len(result):
        a = -.126 * math.log(.001 * len(result))
    else:
        a = 0
    mu = 0
    sigma_sq = len(result)
    
    total_prob = 0
    for ind, move in enumerate(result):
        move.prob = a * math.e**(-(ind+1 - mu)**2 / (2 * sigma_sq))
        total_prob += move.prob
    if len(result) and total_prob < 1:
        result[0].prob += 1 - total_prob

    # print(list(move.prob for move in result))
    
    # # .35, .25, .10, .05
    # rank1 = from_pv
    # for move in rank1: move.prob = .35 / len(rank1)
    # rank2 = list(itertools.chain(cap_see_gt0, checks))
    # for move in rank2: move.prob = .25 * len(rank2) / len(moves)
    # rank3 = list(itertools.chain(counters, killers))
    # for move in rank3: move.prob = .10 * len(rank3) / len(moves)
    # rank4 = other_moves
    # rank5 = cap_see_lt0
    
    return result

def evaluate(position, debug=False):
    # Check for mate
    if position.is_mate():
        # side to move just got mated
        return -1000000
    
    # TODO: implement stalemate
    
    white = black = 0

    # count material
    counts = piece_counts(position)
    for piece_type in [PieceType.P, PieceType.N, PieceType.B,
                       PieceType.R, PieceType.Q, PieceType.K]:
        piece_type_w = PieceType.piece(piece_type.value, Side.WHITE.value)
        piece_type_b = PieceType.piece(piece_type.value, Side.BLACK.value)

        if piece_type is not PieceType.K:
            white += counts[piece_type_w] * material_eval(counts, piece_type.value,
                                                          Side.WHITE.value)
            black += counts[piece_type_b] * material_eval(counts, piece_type.value,
                                                          Side.BLACK.value)

        # positional bonuses and penalties
        if piece_type == PieceType.R:
            for rook in iterate_pieces(position.pieces[piece_type_w]):
                white += rook_position_bonus(rook, position, Side.WHITE.value)
            for rook in iterate_pieces(position.pieces[piece_type_b]):
                black += rook_position_bonus(rook, position, Side.BLACK.value)

        if piece_type in [PieceType.B, PieceType.N]:
            for minor in iterate_pieces(position.pieces[piece_type_w]):
                white += minor_outpost_bonus(piece_type.value, position, Side.WHITE.value)
            for minor in iterate_pieces(position.pieces[piece_type_b]):
                black += minor_outpost_bonus(piece_type.value, position, Side.BLACK.value)

        if piece_type == PieceType.P:
            white += pawn_structure(position, Side.WHITE.value)
            black += pawn_structure(position, Side.BLACK.value)
        
        # piece-square table adjustments
        if piece_type in [PieceType.P, PieceType.N, PieceType.B, PieceType.K]:
            white += psqt_value(piece_type_w, position, Side.WHITE.value)
            black += psqt_value(piece_type_b, position, Side.BLACK.value)
    
    # unprotected
    white += unprotected_penalty(position, Side.WHITE.value)
    black += unprotected_penalty(position, Side.BLACK.value)
            
    # mobility
    white += mobility(position, Side.WHITE.value) * 3
    black += mobility(position, Side.BLACK.value) * 3
            
    # king safety, castle readiness
    if not preserved_castle_rights(position.position_flags, Side.WHITE.value):
        if position.w_king_castle_ply == -1:
            white += -150
    elif white_can_castle_kingside(position.position_flags, position.attacks[Side.BLACK.value], position.occupied[Side.WHITE.value]):
        white += (2 - count_bits(position.occupied[Side.WHITE.value] & (F1 | G1))) * 8
    elif white_can_castle_queenside(position.position_flags, position.attacks[Side.BLACK.value], position.occupied[Side.WHITE.value]):
        white += (3 - count_bits(position.occupied[Side.WHITE.value] & (D1 | C1 | B1))) * 8
    if not preserved_castle_rights(position.position_flags, Side.BLACK.value):
        if position.b_king_castle_ply == -1:
            black += -150
    elif black_can_castle_kingside(position.position_flags, position.attacks[Side.WHITE.value], position.occupied[Side.BLACK.value]):
        black += (2 - count_bits(position.occupied[Side.BLACK.value] & (F8 | G8))) * 8
    elif white_can_castle_queenside(position.position_flags, position.attacks[Side.WHITE.value], position.occupied[Side.BLACK.value]):
        black += (3 - count_bits(position.occupied[Side.BLACK.value] & (D8 | C8 | B8))) * 8
    
    king_zone_w = king_safety_squares(position, Side.WHITE.value)
    king_zone_b = king_safety_squares(position, Side.BLACK.value)
        
    # .. pawn cover of own king
    white += pawn_cover_bonus(king_zone_w, position, Side.WHITE.value)
    black += pawn_cover_bonus(king_zone_b, position, Side.BLACK.value)
    
    # .. king attack bonuses
    white += king_zone_attack_bonus(king_zone_b, position, Side.WHITE.value)
    black += king_zone_attack_bonus(king_zone_w, position, Side.BLACK.value)
    
    res_value = int(white - black)
    if position.white_to_move():
        return res_value
    else:
        return -res_value

def print_moves(moves):
    print(' '.join(map(str,moves)))
