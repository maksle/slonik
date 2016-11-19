from collections import namedtuple
from position import *
from constants import *
from collections import defaultdict
from material import *
from psqt import *
# from evals import eval_see, rook_position_bonus, pawn_cover_bonus, king_safety_squares, king_zone_attack_bonus
from evals import *
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
    def children(self, si=None, depth=0, quiescence=False):
        pv = [] if si is None else si.pv[:]
        moves = self.position.generate_moves()
        smoves = sort_moves(moves, self.position, pv, depth, quiescence)
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
History = namedtuple('History', ['move', 'depth', 'value'])
move_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_history(side, move, depth, value):
    entry = move_history[side][move.piece_type][bit_position(move.to_sq)]
    adjusted_val = value
    if entry:
        adjusted_val = entry.value + .2 * (value - entry.value)
    move_history[side][move.piece_type][bit_position(move.to_sq)] = History(move, depth, adjusted_val)
def lookup_history(side, move):
    if move and move.piece_type is not PieceType.NULL.value:
        return move_history[side][move.piece_type][bit_position(move.to_sq)]
    
# counter moves
Counter = namedtuple('Counter', ['move', 'depth'])
counter_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_counter(side, move, counter, depth):
    entry = counter_history[side][move.piece_type][bit_position(move.to_sq)]
    # adjusted_val = int(value * 100)
    # if entry:
    #     adjusted_val = int(entry.value * .7 + value * 100 * .3)
    counter_history[side][move.piece_type][bit_position(move.to_sq)] = Counter(counter, depth)
def lookup_counter(side, move):
    if move and move.piece_type is not PieceType.NULL.value:
        return counter_history[side][move.piece_type][bit_position(move.to_sq)]
    
tb_hits = 0
node_count = 0
killer_moves = defaultdict(list)

def add_killer(depth, move):
    if move in killer_moves[depth]:
        return
    if len(killer_moves[depth]) == 3:
        killer_moves[depth].pop()
    killer_moves[depth].append(move)

def search(node, si, a, b, depth, pv_node):
    global node_count
    node_count += 1

    assert(pv_node or a == b-1)
    
    a_orig = a
    is_root = si.ply == 0 and pv_node
    
    found = False
    if not pv_node:
        found, tt_ind, tt_entry = tt.get_tt_index(node.position.zobrist_hash)
        if found and tt_entry.depth >= depth:
            if (tt_entry.bound_type == tt.BoundType.LO_BOUND.value and tt_entry.value >= b) \
               or (tt_entry.bound_type == tt.BoundType.HI_BOUND.value and tt_entry.value < a) \
               or tt_entry.bound_type == tt.BoundType.EXACT.value:
                move = Move.move_uncompacted(tt_entry.move)
                return tt_entry.value, move
    
    is_leaf = node.is_leaf()
    if depth == 0 or is_leaf:
        if pv_node:
            si.current_variation = node.position.moves[-si.ply:]
        if is_leaf:
            return node.value(), None
        else:
            score = qsearch(node, si, a, b, depth, pv_node)
            return score, None

    # null move pruning.. if pass move and we're still failing high, dont bother searching
    if not pv_node and not si.null_move_prune_search and node.value() >= b:
        if depth <= 2: nmp_depth = depth
        else: nmp_depth = max(depth-2, 2)

        si.null_move_prune_search = True
        si.ply += 1
        node.position.toggle_side_to_move()
        val, chosen_move = search(node, si, -b, -b+1, nmp_depth, False)
        val = -val
        node.position.toggle_side_to_move()
        si.ply -=1
        si.null_move_prune_search = False

        if val >= b:
            return val, None
        
    best_move = None
    best_move_is_capture = False
    best_val = LOW_BOUND
    move_count = 0
    
    for move in node.children(si=si, depth=depth):
        # print(move)
        # if str(move) == "Qg4-d4":
        #     print("debug")
        child = SearchPos(node.position)
        child.position.make_move(move)
        
        move_count += 1
        extensions = 0
        
        see_score = eval_see(node.position, move)
        is_capture = move.to_sq & node.position.occupied[child.position.side_to_move()]
        gives_check = child.position.in_check()
        
        # extend checks one ply if it's eval_see is not losing
        if gives_check and see_score >= 0:
            extensions = 1
        
        new_depth = depth - 1 + extensions
        r = 0
        if depth > 4: r += 1
        if move_count > 5: r += 1
        if r > 1:
            if is_capture:
                r -= 1 # possible tactics
            elif eval_see(node.position, None, move.from_sq) < 0: 
                r -= 1 # capture evasion
            
        lmr_depth = max(new_depth - r, 0)
        
        # prune moves with too-negative SEE; more lenient with moves closer to root
        if see_score + 250 < -50 * lmr_depth * lmr_depth:
            # print("pruning", see_score, lmr_depth, list(map(str,node.position.moves)), move)
            continue
        
        # TODO?: if non rootNode and we're losing, only look at checks/big captures >= alpha 
        
        if lmr_depth < 3 and not is_capture:
            counter = lookup_counter(node.position.side_to_move(), move)
            history = lookup_history(node.position.side_to_move(), move)
            # TODO: check that counter is legal?
            if counter and history and history.value < -10:
                print("hist val", history.value, "cnter", counter.move, "pruning", move)
                continue
                # lmr_depth = max(lmr_depth - 1, 0)
                
        # Principal Variation Search with Late Move Reductions
        do_full_depth = False
        # LMR null window search
        # TODO: ensure to also check `not is_promo`
        if lmr_depth >= 3 and not is_capture and move_count > 1:
            si.ply += 1
            val, chosen_move = search(child, si, -(a+1), -a, lmr_depth, False)
            val = -val
            si.ply -=1
            # fail high re-search full depth
            if val > a and lmr_depth != new_depth:
                do_full_depth = True
        elif not pv_node or move_count > 1: # pv_node with move_count == 1 will get full window search
            do_full_depth = True
        
        # full depth null window search
        if do_full_depth:
            si.ply += 1
            val, chosen_move = search(child, si, -(a+1), -a, new_depth, False)
            val = -val
            si.ply -=1

        # full window search
        if pv_node and (move_count == 1 or a < val < b):
            si.ply += 1
            val, chosen_move = search(child, si, -b, -a, new_depth, True)
            val = -val
            si.ply -=1
        
        if val > best_val:
            best_val, best_move = val, move
            # if move_count > 5:
            #     print("movecount", move_count, move, list(map(str,node.position.moves)))
            best_move_is_capture = is_capture
            if val > a and pv_node:
                si.pv = si.pv[:si.ply] + si.current_variation[si.ply:]
                if is_root:
                    print_moves(si.pv)
                
        a = max(a, val)
        if a >= b:
            if best_move: 
                add_killer(depth-1, best_move)
            break

    prior_move = node.position.last_move()
    if best_move:
        if not best_move_is_capture:
            bonus = (depth * depth) + (2 * depth) - 2
            update_history(node.position.side_to_move(), best_move, depth, bonus)
            if prior_move and prior_move.piece_type != PieceType.NULL.value:
                update_counter(node.position.side_to_move() ^ 1, prior_move, best_move, depth)
                # penalize prior move that allowed this good move
                # TODO: make sure that move was a quiet move?
                if len(node.position.moves) > 1 and node.position.moves[-2] != PieceType.NULL.value:
                    update_history(node.position.side_to_move(), node.position.moves[-2], depth, -(bonus + 4))
    else:
        # reward the move that caused this node to fail low
        # TODO: make sure node.moves[-2] was a quiet move?
        # TODO: maybe don't reward at too shallow depth (less confident of it)
        bonus = (depth * depth) + (2 * depth) - 2
        if len(node.position.moves) > 1 and node.position.moves[-2] != PieceType.NULL.value:
            update_history(node.position.side_to_move(), node.position.moves[-2], depth, bonus)
        
    if not found:
        tt_entry = tt.TTEntry()
        tt_entry.key = node.position.zobrist_hash
    tt_entry.value = best_val
    tt_entry.depth = depth
    if best_move is not None:
        chosen_move = best_move.compact()
    else:
        chosen_move = 0
    tt_entry.move = chosen_move
    if best_val <= a_orig:
        bound_type = tt.BoundType.HI_BOUND.value
    elif best_val >= b:
        bound_type = tt.BoundType.LO_BOUND.value
    else:
        bound_type = tt.BoundType.EXACT.value
    tt_entry.bound_type = bound_type
    tt.save_tt_entry(tt_entry)
    
    return best_val, best_move

def qsearch(node, si, alpha, beta, depth, pv_node):
    global node_count
    assert(pv_node or alpha == beta-1)
    assert(depth <= 0)
    node_count += 1

    tt_hit = False
    a_orig = alpha
    
    if not pv_node:
        tt_hit, tt_ind, tt_entry = tt.get_tt_index(node.position.zobrist_hash)
        if tt_hit and tt_entry.depth <= 0:
            if tt_entry.bound_type == tt.BoundType.EXACT.value:
                return tt_entry.value
            
            if tt_entry.bound_type == tt.BoundType.LO_BOUND.value and tt_entry.value >= beta:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.bound_type == tt.BoundType.HI_BOUND.value and tt_entry.value < alpha:
                beta = min(beta, tt_entry.value)

            if alpha >= beta:
                return tt_entry.value

    in_check = node.position.in_check()
            
    # if in_check:
    #     best_value = start_val = LOW_BOUND
    # else:
    #     best_value = start_val = node.value()
    best_value = start_val = node.value()
    if best_value >= beta:
        # "stand pat"
        if not tt_hit:
            tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash,
                                        Move(PieceType.NULL.value, None, None).compact(),
                                        tt.BoundType.LO_BOUND.value, best_value, depth))
            return best_value
    
    if pv_node and best_value > alpha:
        alpha = best_value

    best_move = Move(PieceType.NULL.value, None, None)
    score = None
        
    for move in node.children(si=si, quiescence=True):
        # print_moves(node.children(si=si, quiescence=True))
        if ' '.join(map(str,node.position.moves)) == "Nc6-d4 Nf3-d4 Qg4-d4 Bd3-b5":
        # if ' '.join(map(str,node.position.moves)) == "Nc6-d4 Nf3-d4":
            print("debug")
        # only evaluate captures with SEE >= 0, assumes there is always a better move
        # TODO?: only prune this if node is not in check
        # TODO?: only prune this if move is not a promotion, or take promo in account in eval_see

        child = SearchPos(node.position)
        child.position.make_move(move)
            
        gives_check = child.position.in_check()
        
        # is_capture = move.to_sq & node.position.occupied[child.position.side_to_move()]
        if not in_check and not gives_check:
            static_eval = eval_see(node.position, move)
            if static_eval < 0:
                continue
            # Futility pruning: losing and move doesn't overtake alpha my a large margin
            if start_val + static_eval + MG_PIECES[PieceType.P.value] <= alpha:
                continue

        score = -qsearch(child, si, -beta, -alpha, depth-1, pv_node)
        
        # if str(move) == "Qd1-d4":
        #     print("debug")
        
        if score > best_value:
            best_value = score
            if score > alpha:
                if pv_node and score < beta:
                    alpha = score
                    best_move = move
                else:
                    assert score >= beta
                    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, move.compact(),
                                                tt.BoundType.LO_BOUND.value, best_value, depth))
                    return score
        
    # TODO: check if we're mated
    if pv_node and best_value > a_orig:
        bound_type = tt.BoundType.EXACT.value
    else:
        bound_type = tt.BoundType.HI_BOUND.value
        # if best_move.piece_type == PieceType.NULL.value and score is not None and node.position.in_check():
        #     best_value = score
    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, best_move.compact(),
                                bound_type, best_value, depth))
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

def iterative_deepening(target_depth, node):
    alpha = LOW_BOUND
    beta = HIGH_BOUND

    depth = 0
    val = 0

    si = SearchInfo()
    
    while depth != target_depth:
        depth += 1
        finished = False
        fail_factor = 18

        alpha, beta = val - fail_factor, val + fail_factor
        
        print (">> depth", depth)
        
        while not finished:
            val, chosen = search(node, si, alpha, beta, depth, True)

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

        # print_moves(find_pv(node.position))
        # print_moves(final.position.moves)

    # import pprint
    # pprint.pprint(move_history)
        
    global node_count
    print("node count", node_count)
    return val, chosen

def sort_moves(moves, position, pv, depth, quiescence):
    from_pv = []
    captures = []
    killers = []
    counters = []
    history = []
    other_moves = []
    checks = []

    side = position.side_to_move()
    other = position.occupied[side ^ 1]
    counter = lookup_counter(side ^ 1, position.last_move())

    def hist_val(move):
        entry = lookup_history(side, move)
        if entry:
            return entry.value
        return 0
    
    for move in moves:
        if quiescence:
            if is_capture(move.to_sq, other):
                captures.append(move)
            else:
                in_check = position.in_check()
                try_move = SearchPos(position)
                try_move.position.make_move(move)
                if in_check and not try_move.position.in_check(side):
                    # evade check
                    other_moves.append(move)
                elif not in_check and try_move.position.in_check():
                    # give check
                    checks.append(move)
        else:
            if move in pv:
                from_pv.append(move)
            elif is_capture(move.to_sq, other):
                captures.append(move)
            elif counter and counter.move == move:
                counters.append(move)
            elif move in killer_moves[depth]:
                killers.append(move)
            else:
                other_moves.append(move)

    captures = sorted(captures, key=lambda c: eval_see(position, c), reverse=True)
    captures_see = map(lambda c: (eval_see(position, c), c), captures)
    sorted_cap_see = sorted(captures_see, key=lambda cs: cs[0], reverse=True)

    if quiescence:
        return itertools.chain(map(lambda c: c[1], sorted_cap_see), checks, other_moves)

    cap_see_gt0 = []
    cap_see_lte0 = []
    for cs in sorted_cap_see:
        if cs[0] > 0:
            cap_see_gt0.append(cs[1])
        else:
            cap_see_lte0.append(cs[1])
    
    # counters = [c.move for c in sorted(counters, key=lambda c: c.value, reverse=True)]
    other_moves = [m for m in sorted(other_moves, key=hist_val, reverse=True)]
    
    return itertools.chain(from_pv, cap_see_gt0, counters, killers, cap_see_lte0, other_moves)

def evaluate(position, debug=False):
    # Check for mate
    if position.is_mate():
        if position.side_to_move() == Side.WHITE.value:
            # white just got mated
            return -1000000
        else:
            return 1000000

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
            
    # king safety
    if not preserved_castle_rights(position.position_flags, Side.WHITE.value) \
       and position.w_king_castle_ply == -1:
        white += -75
    if not preserved_castle_rights(position.position_flags, Side.BLACK.value) \
       and position.b_king_castle_ply == -1:
        black += -75

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
