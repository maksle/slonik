from IPython import embed
import time 
import pprint
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
        self.skip_early_pruning = False
        self.singular_search = False
        self.excluded_move = Move(PieceType.NULL)
        
# history moves
History = namedtuple('History', ['move', 'value'])
move_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_history(side, move, value):
    print(side, move, value)
    entry = move_history[side][move.piece_type][bit_position(move.to_sq)]
    adjusted_val = value
    if entry:
        adjusted_val = entry.value + .2 * (value - entry.value)
    move_history[side][move.piece_type][bit_position(move.to_sq)] = History(move, adjusted_val)
def lookup_history(side, move):
    if move and move.piece_type is not PieceType.NULL:
        return move_history[side][move.piece_type][bit_position(move.to_sq)]

# counter moves
Counter = namedtuple('Counter', ['move'])
counter_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_counter(side, move, counter):
    entry = counter_history[side][move.piece_type][bit_position(move.to_sq)]
    counter_history[side][move.piece_type][bit_position(move.to_sq)] = Counter(counter)
def lookup_counter(side, move):
    if move and move.piece_type is not PieceType.NULL:
        return counter_history[side][move.piece_type][bit_position(move.to_sq)]

def make_move(move):
    for i in range(2):
        for j in range(13):
            for k in range(64):
                entry = move_history[i][j][k] 
                if entry is not None:
                    move_history[i][j][k] = History(entry.move, entry.value)
    # for side in move_history:
    #     for pc in side:
    #         for sq in pc:
    #             if sq is not None:
    #                 embed()
    #                 sq.value /= 2
                
tb_hits = 0
node_count = 0
killer_moves = defaultdict(list)

ply_count = 0
ply_iter = 0
ply_max = 0
def update_ply_stat(ply):
    global ply_count
    global ply_iter
    global ply_max
    ply_count += ply
    ply_iter += 1
    if ply > ply_max: ply_max = ply
    
def add_killer(ply, move):
    if move in killer_moves[ply]:
        return
    if len(killer_moves[ply]) == 3:
        killer_moves[ply].pop()
    killer_moves[ply].append(move)

def depth_to_allowance(depth):
    return int(math.ceil(.193006 * math.e ** (1.29829 * depth)) + depth * 3)

def allowance_to_depth(allowance):
    base = allowance / .193006
    if base <= 0: return 0
    return round(math.log(base) / 1.29829, 1)

def iterative_deepening(target_depth, node, si=None):
    alpha = LOW_BOUND
    beta = HIGH_BOUND

    depth = 0
    
    val = 0
    si = si or [None] * 64
    
    while depth < target_depth:
        depth += .5
        allowance = depth_to_allowance(depth)
        finished = False
        fail_factor = 18

        alpha, beta = val - fail_factor, val + fail_factor
        
        print ("\n>> depth:", depth, ", allowance:", allowance)
        
        now = time.time()
        while not finished:
            val = search(node, si, 0, alpha, beta, allowance, True, False)

            if val < alpha:
                print("faillow", "a", alpha, "b", beta, "val", val, "factor", fail_factor)
            if val > beta:
                print("failhigh", "a", alpha, "b", beta, "val", val, "factor", fail_factor)

            if val <= alpha:
                alpha = val - fail_factor
                # beta = (alpha + beta) // 2
                fail_factor += fail_factor // 3 + 6
            elif val >= beta:
                # alpha = (alpha + beta) / 2
                beta = val + fail_factor
                fail_factor += fail_factor // 3 + 6
            else:
                finished = True
        then = time.time()
        print(then-now, 's')
    
    # pprint.pprint(move_history)
        
    return val, si
    
def search(node, si, ply, a, b, allowance, pv_node, cut_node):
    global node_count
    global ply_iter
    global ply_count
    node_count += 1
    
    assert(pv_node or a == b-1)
    
    a_orig = a
    is_root = pv_node and ply == 0
    
    si[ply] = si[ply] or SearchInfo()
    si[ply+1] = si[ply+1] or SearchInfo()

    si[ply].pv.clear()
    si[ply+1].pv.clear()
    
    pos_key = node.zobrist_hash ^ si[ply].excluded_move.compact()
    
    found = False
    found, tt_ind, tt_entry = tt.get_tt_index(pos_key)
    if not pv_node and found and tt_entry.depth >= allowance:
        if (tt_entry.bound_type == tt.BoundType.LO_BOUND and tt_entry.value >= b) \
           or (tt_entry.bound_type == tt.BoundType.HI_BOUND and tt_entry.value < a) \
           or tt_entry.bound_type == tt.BoundType.EXACT:
            return tt_entry.value

    if found and tt_entry.static_eval is not None:
        si[ply].static_eval = static_eval = tt_entry.static_eval
    else:
        if node.last_move() == Move(PieceType.NULL):
            si[ply].static_eval = static_eval = -si[ply-1].static_eval + 40
        else:
            si[ply].static_eval = static_eval = Evaluation(node).init_attacks().evaluate()
        tt.save_tt_entry(tt.TTEntry(pos_key, 0, tt.BoundType.NONE, 0, 0, static_eval))
        
    in_check = node.in_check()
    
    if allowance_to_depth(allowance) < 1:
        score = qsearch(node, si, ply, a, b, 9, pv_node, in_check)
        return score
    
    if not in_check:
        if not si[ply].skip_early_pruning:
            # futility prune of parent
            if allowance_to_depth(allowance) < 6 \
               and static_eval - allowance_to_depth(allowance) * 150 >= b:
                return static_eval
            
            # null move pruning.. if pass move and we're still failing high, dont bother searching further
            if not pv_node and not si[ply].null_move_prune_search \
               and found and tt_entry.value >= b:
                si[ply+1].null_move_prune_search = True
                node.make_null_move()
                val = -search(node, si, ply+1, -b, -b+1, int(allowance * .75), False, False)
                node.undo_null_move()
                si[ply+1].null_move_prune_search = False

                if val >= b:
                    # print(" **** NULL BETA CUTTOFF after", ' '.join(map(str,node.moves)), "--winning for", node.side_to_move(), tt_entry.value, val, a, b)
                    return val

        # internal iterative deepening to improve move order when there's no pv
        if not found and allowance_to_depth(allowance) >= 4 \
           and (pv_node or static_eval + MG_PIECES[PieceType.P] >= b):
            si[ply+1].skip_early_pruning = True
            val = search(node, si, ply, a, b, int(allowance * .75), pv_node, cut_node)
            si[ply+1].skip_early_pruning = False
            found, tt_ind, tt_entry = tt.get_tt_index(node.zobrist_hash)
    
    singular_extension_eligible = False
    if not is_root and allowance_to_depth(allowance) >= 2.5 \
       and not si[ply].singular_search \
       and found \
       and tt_entry.move != 0 \
       and tt_entry.bound_type != tt.BoundType.HI_BOUND \
       and tt_entry.depth >= allowance * .7:
        singular_extension_eligible = True

    improving = ply < 2 or si[ply-2].static_eval is None or si[ply].static_eval >= si[ply-2].static_eval
        
    best_move = None
    best_move_is_capture = False
    best_val = val = LOW_BOUND
    move_count = 0

    next_move_prob = None
    
    if in_check:
        pseudo_moves = node.generate_moves_in_check()
    else:
        pseudo_moves = node.generate_moves_all()
    moves = sort_moves(pseudo_moves, node, si, ply, False)
    
    counter = None
    if len(node.moves):
        counter = lookup_counter(node.side_to_move() ^ 1, node.moves[-1])
    
    for move in moves:
        # if ' '.join(map(str,node.moves)) == "Ng4-e5 Nf3-e5 Nc6-e5 Rf1-e1 f7-f6 f2-f4" and str(move) == "f6-f5":
        #     print("debug")
    
        if counter is not None and counter.move == move:
            move.prob *= 1.25
        elif not improving:
            move.prob *= .75
        
        if si[ply].excluded_move == move:
            continue
        
        if not node.is_legal(move):
            continue
        
        child = Position(node)
        child.make_move(move)
        move_count += 1

        gives_check = child.in_check()
        if gives_check:
            move.move_type |= MoveType.check
        
        is_capture = move.to_sq & node.occupied[child.side_to_move()]
        see_score = move.see_score or eval_see(node, move)

        extending = False

        if gives_check and move.prob > .01 and see_score > 0:
            extending = True
            move.prob = 1
        
        if not extending and pv_node and len(node.moves) > 0 and node.moves[-1] != Move(PieceType.NULL):
            # recapture extension
            if node.moves[-1].move_type & MoveType.capture and node.moves[-1].to_sq == move.to_sq:
               extending = True
               move.prob = 1
            
        # singular extension logic pretty much as in stockfish
        if not extending and singular_extension_eligible and move == Move.move_uncompacted(tt_entry.move):
            # print("Trying Singular:", ' '.join(map(str,node.moves)), move)
            rbeta = int(tt_entry.value - (2 * allowance_to_depth(allowance)))
            si[ply].excluded_move = move
            si[ply].singular_search = True
            si[ply].skip_early_pruning = True
            val = search(node, si, ply, rbeta-1, rbeta, int(allowance * move.prob * .75), False, cut_node)
            si[ply].skip_early_pruning = False
            si[ply].singular_search = False
            si[ply].excluded_move = Move(PieceType.NULL)
            # print("val", val, "rbeta", rbeta)
            
            if val < rbeta:
                extending = True
                move.prob = 1

        if not is_root and not is_capture and not gives_check:
            next_depth = allowance_to_depth(allowance * move.prob)
            if next_depth <= 5 and not in_check \
               and static_eval + 1.25 * MG_PIECES[PieceType.P] + next_depth * MG_PIECES[PieceType.P] <= a:
                continue
        
            # prune moves with too-negative SEE; more lenient with moves closer to root
            if next_depth < 8 and see_score <= -35 * next_depth ** 2:
                continue
        
        # TODO?: if non rootNode and we're losing, only look at checks/big captures >= alpha 
        
        # Probabilistic version of LMR
        # .. zero window search reduced 
        zw_allowance = allowance * move.prob 
        if allowance_to_depth(allowance) >= 3 and not is_capture:
            r = min(zw_allowance * .25, depth_to_allowance(1))
            if cut_node:
                r += depth_to_allowance(2)
            else:
                child.make_null_move()
                undo_see = eval_see(child, Move(move.piece_type, move.to_sq, move.from_sq))
                child.undo_null_move()
                if undo_see < 0:
                    # reduce reduction if escaping capture
                    r -= depth_to_allowance(2)
                else:
                    # reduce reduction if making a threat
                    ep_default_value = (0, 0, 0, 0)
                    victim_before, *rest = next_en_prise(node, child.side_to_move())
                    victim_after, *rest = next_en_prise(child, child.side_to_move())
                    if victim_after > victim_before:
                        r -= depth_to_allowance(2)

            hist = lookup_history(node.side_to_move(), move)
            # print("hist", move, hist)
            if hist and hist.value > 20:
                r -= depth_to_allowance(1)
            elif hist and hist.value < -20:
                r += depth_to_allowance(1)

            if r < 0: r = 0

            val = -search(child, si, ply+1, -(a+1), -a, int(zw_allowance - r), False, True)
            do_full_zw = val > a and r != 0
        else:
            do_full_zw = not (pv_node and move.prob >= .4)
        
        # .. zero window full allotment search
        if do_full_zw:
            val = -search(child, si, ply+1, -(a+1), -a, int(zw_allowance), True, not cut_node)
            
        # .. full window full allotment search
        # otherwise we let the fail highs cause parent to fail low and try different move
        if pv_node and (move.prob >= .4 or (a < val and (val < b or is_root))):
            val = -search(child, si, ply+1, -b, -a, int(allowance * move.prob), True, False)

        if val > best_val:
            best_val, best_move = val, move
            # if move_count > 5:
            #     print("movecount", move_count, move, list(map(str,node.moves)))
            best_move_is_capture = is_capture
            if pv_node:
                print("local best move", node.moves, move, "val:", val, "alpha", a_orig, "beta", b)
                # if is_root:
                #     print("new best move", move, "val:", val, "alpha", a_orig, "beta", b)
                if val > a:
                    # next_move_prob = None
                    si[ply].pv = [move] + si[ply+1].pv
                    si[ply+1].pv = []
                    if is_root:
                        print_moves(si[ply].pv)
                        print("new best move (root,>a)", move, "val:", val, "alpha", a_orig, "beta", b)
                
        a = max(a, val)
        # print(node.moves, move)
        if a >= b:
            print("fail-high", node.moves, "\"", best_move, "\"", val, a_orig, b)
            if best_move: 
                add_killer(ply, best_move)
            break
        elif a <= a_orig:
            print("fail-low", node.moves, "\"", move, "\"", val, a_orig, b)
        
    prior_move = node.last_move()
    if move_count == 0:
        if si[ply].excluded_move != Move(PieceType.NULL): best_val = a
        # mate or statemate
        elif in_check: best_val = MATE_VALUE
        else: best_val = DRAW_VALUE
    elif best_move:
        if not best_move_is_capture:
            bonus = int(allowance) ** 2
            update_history(node.side_to_move(), best_move, bonus)
            if prior_move and prior_move.piece_type != PieceType.NULL:
                update_counter(node.side_to_move() ^ 1, prior_move, best_move)
                # penalize prior quiet move that allowed this good move
                if len(node.moves) > 1 and prior_move.piece_type != PieceType.NULL and not prior_move.move_type & MoveType.capture:
                    update_history(node.side_to_move() ^ 1, prior_move, -(bonus + 4))
    elif allowance_to_depth(allowance) >= 2.5 and not best_move_is_capture:
        assert(best_val <= a)
        # reward the quiet move that caused this node to fail low
        bonus = int(allowance) ** 2
        if len(node.moves) > 1 and prior_move.piece_type != PieceType.NULL and not prior_move.move_type & MoveType.capture:
            update_history(node.side_to_move() ^ 1, prior_move, bonus)

    # if best_val == LOW_BOUND:
    #     print("best_val is LOW_BOUND, node.moves", node.moves, best_val)
    # assert(best_val > LOW_BOUND)
    
    best_move = best_move or Move(PieceType.NULL)
    if best_val <= a_orig: bound_type = tt.BoundType.HI_BOUND
    elif best_val >= b: bound_type = tt.BoundType.LO_BOUND
    else: bound_type = tt.BoundType.EXACT
    tt.save_tt_entry(tt.TTEntry(pos_key, best_move.compact(),
                                bound_type, best_val, allowance, static_eval))

    if is_root:
        real_pv = a_orig < best_val < b
        print("node_count:", node_count, "avg ply:", ply_count / (ply_iter or 1), "max ply:", ply_max, "alpha", a, "beta", b, "val", best_val)
        print("REAL PV:" if real_pv else "pv:", si[ply].pv)
        print()

    return best_val

def qsearch(node, si, ply, alpha, beta, allowance, pv_node, in_check):
    global node_count
    assert(pv_node or alpha == beta-1)
    node_count += 1

    if pv_node:
        si[ply] = SearchInfo()
        si[ply+1] = SearchInfo()

    tt_hit = False
    a_orig = alpha

    # if ' '.join(map(str, node.moves)) == 'e2-e4 e7-e6 Qd1-f3':
    #     print("debug")
    
    tt_hit, tt_ind, tt_entry = tt.get_tt_index(node.zobrist_hash)
    if not pv_node:
        if tt_hit and tt_entry.depth <= allowance:
            if tt_entry.bound_type == tt.BoundType.EXACT:
                return tt_entry.value
            
            if tt_entry.bound_type == tt.BoundType.LO_BOUND and tt_entry.value >= beta:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.bound_type == tt.BoundType.HI_BOUND and tt_entry.value < alpha:
                beta = min(beta, tt_entry.value)

            if alpha >= beta:
                update_ply_stat(ply)
                # print(node.moves, tt_entry.value)
                return tt_entry.value
    
    # to stop search of endless checks, including repetition checks
    if in_check:
        num_moves = len(node.moves)
        if allowance <= 1 and num_moves > 3 \
           and node.moves[-3].move_type & MoveType.check:
            update_ply_stat(ply)
            static_eval = Evaluation(node).init_attacks().evaluate()
            # print(node.moves, static_eval)
            return static_eval
    
    static_eval = None
    if tt_hit:
        static_eval = tt_entry.static_eval
    
    best_move = Move(PieceType.NULL)
    if in_check:
        best_value = start_val = LOW_BOUND
    else:
        if static_eval is None:
            best_value = start_val = static_eval = Evaluation(node).init_attacks().evaluate()
        else:
            best_value = start_val = static_eval

        if best_value >= beta:
            # "stand pat"
            if not tt_hit or tt_entry.bound_type == tt.BoundType.NONE:
                tt.save_tt_entry(tt.TTEntry(node.zobrist_hash,
                                        Move(PieceType.NULL, None, None).compact(),
                                        tt.BoundType.LO_BOUND, best_value, allowance, static_eval))
            update_ply_stat(ply)
            # print(node.moves, best_value)
            return best_value

        if pv_node and best_value > alpha:
            alpha = best_value

    score = None
    move_count = 0

    if in_check:
        pseudo_moves = node.generate_moves_in_check()
    else:
        pseudo_moves = node.generate_moves_violent()
    moves = sort_moves(pseudo_moves, node, si, ply, True)
    for move in moves:
        # if ' '.join(map(str,node.moves)) == "Ng4-e5 Nf3-e5 Nc6-e5 Bc4-f7 Ke8-e7" and str(move) == "Rf1-e1":
        #     print("debug")
        
        if not node.is_legal(move):
            continue
        
        child = Position(node)
        child.make_move(move)
        move_count += 1

        gives_check = child.in_check()
        if gives_check:
            move.move_type |= MoveType.check

        is_capture = move.to_sq & node.occupied[node.side_to_move() ^ 1]
        
        if not in_check and not gives_check:
            # Futility pruning
            # .. try to avoid calling eval_see
            pt_captured = node.squares[bit_position(move.to_sq)]
            if start_val + MG_PIECES[PieceType.base_type(pt_captured)] + MG_PIECES[PieceType.P] <= alpha:
                continue
            see_score = move.see_score if move.see_score is not None else eval_see(node, move)
            if start_val + see_score + MG_PIECES[PieceType.P] <= alpha \
               and see_score < 0:
                continue

        if not in_check or not is_capture:
            see_score = move.see_score if move.see_score is not None else eval_see(node, move)
            if see_score < 0:
                continue
        
        score = -qsearch(child, si, ply+1, -beta, -alpha, int(allowance * move.prob), pv_node, gives_check)
        
        if score > best_value:
            best_value = score
            if score > alpha:
                if pv_node and score < beta:
                    alpha = score
                    best_move = move
                    si[ply].pv = [move] + si[ply+1].pv
                    si[ply+1].pv = []
                else:
                    # assert score >= beta
                    if score >= beta:
                        tt.save_tt_entry(tt.TTEntry(node.zobrist_hash, move.compact(),
                                                    tt.BoundType.LO_BOUND, best_value, allowance, static_eval))
                        update_ply_stat(ply)
                        # print(node.moves, score)
                        return score

        # print("q", node.moves, move)
    
    if move_count == 0:
        if in_check and best_value == LOW_BOUND:
            return MATE_VALUE
        # can't assume it's stalemate here since not generating all legal moves in qsearch
        if static_eval is None:
            static_eval = Evaluation(node).init_attacks().evaluate()
        best_value = static_eval
        
    if pv_node and best_value > a_orig: bound_type = tt.BoundType.EXACT
    else: bound_type = tt.BoundType.HI_BOUND
    tt.save_tt_entry(tt.TTEntry(node.zobrist_hash, best_move.compact(),
                                bound_type, best_value, allowance, static_eval))

    update_ply_stat(ply)
    return best_value

def find_pv(root_pos):
    moves = []
    pos = Position(root_pos)
    
    def find_next_move():
        found, tt_ind, tt_entry = tt.get_tt_index(pos.zobrist_hash)
        if found:
            move = Move.move_uncompacted(tt_entry.move)
            if move != 0 and tt_entry.bound_type in [tt.BoundType.EXACT, tt.BoundType.LO_BOUND]:
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
    us, them = side, side ^ 1
    
    other = position.occupied[them]
    counter = lookup_counter(them, position.last_move())
    
    ep_us_before = next_en_prise(position, us)
    ep_them_before = next_en_prise(position, them)
    
    def sort_crit(move, en_prise_sort=False):
        entry = lookup_history(us, move)
        see_val = eval_see(position, move)

        if en_prise_sort:
            ep_us_after = next_en_prise(position, us, move)
            pt, *rest = ep_us_before
            pt2, *rest2 = ep_us_after
            loss = pt - pt2
            see_val += loss

            ep_them_after = next_en_prise(position, them, move)
            pto, *rest = ep_them_before
            pto2, *rest = ep_them_after
            gain = pto2 - pto
            see_val += gain
        
        hist_val = entry.value if entry else 0
        psqt_val = psqt_value_sq(move.piece_type, move.to_sq, us)
        return (see_val, hist_val, psqt_val)
    
    for move in moves:
        if quiescence:
            if move.move_type & MoveType.capture:
                captures.append(move)
            else:
                other_moves.append(move)
        else:
            if move in find_pv(position):
                from_pv.append(move)
            elif is_capture(move.to_sq, other):
                captures.append(move)
            elif move.move_type == MoveType.check:
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
        result = list(itertools.chain(map(itemgetter(1), sorted_cap_see), captures, other_moves))
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
        result = list(itertools.chain(from_pv, cap_see_gt0, cap_see_eq0, checks, counters, killers, other_moves, cap_see_lt0))

    # gaussian formula
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
    
    # print([move.prob for move in result])
    return result

def print_moves(moves):
    print(' '.join(map(str,moves)))

def perft_outer(depth):
    si = [None] * 64
    return perft(depth, si)

def perft(pos, depth, is_root):
    cnt = nodes = 0
    leaf = depth == 2
    moves = list(pos.generate_moves())
    for move in moves:
        if is_root and depth <= 1:
            cnt = 1
            nodes += 1
        else:
            child = Position(pos)
            child.make_move(move)
            if leaf:
                cnt = len(list(child.generate_moves())) 
            else:
                cnt = perft(child, depth - 1, False)
            nodes += cnt
        if is_root:
            print("move:", str(move), cnt)
    return nodes

def evaluate_moves(position):
    moves = list(position.generate_moves())
    scores = []
    for move in moves:
        pos = Position(position)
        pos.make_move(move)
        scores.append(Evaluation(pos).init_attacks().evaluate())
    res = list(sorted(zip(map(str,moves), scores), key=lambda x: x[1]))
    return pprint.pprint(res)

def discoveries_and_pins(position, side, target_piece_type=Pt.K):
    """Return squares of singular pieces between sliders and the king of side
    `side`. Blockers of opposite side can move and cause discovered check, and
    blockers of same side are pinned. `target_piece_type` is piece things are pinned to."""
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

    pinned = 0
    discoverers = 0
    for sniper_sq in iterate_pieces(snipers):
        # queen vs queen and the target piece is defended, assume not pin (ignoring possibility of more xrays)
        if Pt.base_type(position.squares[bit_position(sniper_sq)]) == target_piece_type \
           and position.attacks[side] & target_piece_sq:
            continue
        squares_between = BETWEEN_SQS[bit_position(sniper_sq)][bit_position(target_piece_sq)]
        if count_bits(squares_between & all_occupied) == 1:
            p = us & squares_between
            d = them & squares_between
            if p:
                pinned |= p
            elif d and Pt.base_type(position.squares[bit_position(d)]) != Pt.P:
                discoverers |= d

    # position.pinned[Pt.piece(target_piece_type, side)] = pinned
    return (discoverers, pinned)
