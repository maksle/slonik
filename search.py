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

class SearchPos():
    def __init__(self, position):
        self.position = Position(position)
    def is_leaf(self):
        return self.position.is_mate()
    def value(self):
        return evaluate(self.position)
    def children(self, si=None, depth=0, captures_only=False):
        pv = [] if si is None else si.pv[:]
        moves = self.position.generate_moves()
        if captures_only:
            side = self.position.side_to_move()
            moves = [move for move in moves
                     if move.to_sq & self.position.occupied[side ^ 1]]
        smoves = sort_moves(moves, self.position, pv, depth-1)
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
        adjusted_val = .8 * entry.value + .2 * (value - entry.value)
    move_history[side][move.piece_type][bit_position(move.to_sq)] = History(move, depth, adjusted_val)
def lookup_history(side, move):
    if move and move.piece_type is not PieceType.NULL.value:
        return counter_history[side][move.piece_type][bit_position(move.to_sq)]
    
# counter moves
Counter = namedtuple('Counter', ['move', 'depth', 'value'])
counter_history = [[[None for i in range(64)] for i in range(13)] for i in range(2)]
def update_counter(side, move, counter, depth, value):
    entry = counter_history[side][move.piece_type][bit_position(move.to_sq)]
    adjusted_val = int(value * 100)
    if entry:
        adjusted_val = int(entry.value * .7 + value * 100 * .3)
    counter_history[side][move.piece_type][bit_position(move.to_sq)] = Counter(counter, depth, adjusted_val)
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
            print("mate:", end="")
            print_moves(node.position.moves)
            return node.value(), None
        else:
            score = qsearch(node, si, a, b, depth, pv_node)
            return score, None

    best_move = None
    best_move_is_capture = False
    best_val = LOW_BOUND
    move_count = 0
    
    for move in node.children(si=si, depth=depth):
        # if str(node.position.last_move()) == 'Bd2-c1' and move_count == 0:
        #     print("BC1 respnose order", end=" ")
        #     print_moves(node.children(si=si, depth=depth))
        #     counter = lookup_counter(node.position.side_to_move()^1, node.position.last_move())
        #     if counter:
        #         print("counter to bc1:", counter.move)
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
        lmr_depth = max(new_depth - 2, 0)
            
        # prune moves with too-negative SEE; more lenient with moves closer to root
        if see_score < -40 * lmr_depth * lmr_depth:
            continue
        
        # TODO?: if non rootNode and we're losing, only look at checks/big captures >= alpha 

        # add counter moves and if counter move is too good prune the move at shallow depths
        # if lmr_depth < 3:
        #     counter = lookup_counter(node.position.side_to_move(), move)
        #     if counter and counter.value > 100:
        #         print("cutoff at depth", depth)
        #         continue

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

    if best_move and not best_move_is_capture:
        bonus = (depth * depth) + (2 * depth) - 2
        update_history(node.position.side_to_move(), best_move, depth, bonus)
        prior_move = node.position.last_move()
        if prior_move and prior_move.piece_type != PieceType.NULL.value:
            update_counter(node.position.side_to_move() ^ 1, prior_move, best_move, depth, bonus)
            
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
        
        print ("DEPTH", depth, "-----------------------------------------------------")
        
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

def sort_moves(moves, position, pv, depth):
    from_pv = []
    captures = []
    killers = []
    counters = []
    history = []
    other_moves = []
    other = position.occupied[side_to_move(position.position_flags)^1]

    def get_counter(move):
        counter = lookup_counter(position.side_to_move() ^ 1, position.last_move())
        if counter and counter.value > 0 and counter.move == move:
            return counter
        
    for move in moves:
        counter = get_counter(move)
        if move in pv:
            from_pv.append(move)
        elif is_capture(move.to_sq, other):
            captures.append(move)
        elif counter and counter.value > 0:
            counters.append(counter)
        elif move in killer_moves[depth]:
            killers.append(move)
        else:
            other_moves.append(move)

    def hist_val(move):
        entry = lookup_history(position.side_to_move(), move)
        if entry:
            return entry.value
        return 0
    
    captures = sorted(captures, key=lambda c: eval_see(position, c), reverse=True)
    captures_see = map(lambda c: (eval_see(position, c), c), captures)
    sorted_cap_see = sorted(captures_see, key=lambda cs: cs[0], reverse=True)
    cap_see_gt0 = []
    cap_see_lte0 = []
    for cs in sorted_cap_see:
        if cs[0] > 0:
            cap_see_gt0.append(cs[1])
        else:
            cap_see_lte0.append(cs[1])
    
    counters = [c.move for c in sorted(counters, key=lambda c: c.value, reverse=True)]
    other_moves = [m for m in sorted(other_moves, key=hist_val, reverse=True)]
    
    return itertools.chain(from_pv, cap_see_gt0, counters, killers, cap_see_lte0, other_moves)

def evaluate(position):
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

        # positional bonuses
        if piece_type == PieceType.R:
            for rook in iterate_pieces(position.pieces[piece_type_w]):
                white += rook_position_bonus(rook, position, Side.WHITE.value)
            for rook in iterate_pieces(position.pieces[piece_type_b]):
                black += rook_position_bonus(rook, position, Side.BLACK.value)
        
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
        white -= -75
    if not preserved_castle_rights(position.position_flags, Side.BLACK.value) \
       and position.b_king_castle_ply == -1:
        black -= -75

    king_zone_w = king_safety_squares(position, Side.WHITE.value)
    king_zone_b = king_safety_squares(position, Side.BLACK.value)
        
    # .. pawn cover of own king
    white += pawn_cover_bonus(king_zone_w, position, Side.WHITE.value)
    black += pawn_cover_bonus(king_zone_b, position, Side.BLACK.value)
    
    # .. king attack bonuses
    white += king_zone_attack_bonus(king_zone_w, position, Side.WHITE.value)
    black += king_zone_attack_bonus(king_zone_b, position, Side.BLACK.value)
    
    res_value = int(white - black)
    if position.white_to_move():
        return res_value
    else:
        return -res_value

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
        
    for move in node.children(si=si, captures_only=True):
        # only evaluate captures with SEE >= 0, assumes there is always a better move
        # TODO?: only prune this if node is not in check
        # TODO?: only prune this if move is not a promotion, or take promo in account in eval_see
        static_eval = eval_see(node.position, move)
        if static_eval < 0:
            continue

        # Futility pruning: losing and move doesn't overtake alpha my a large margin
        if start_val + static_eval + MG_PIECES[PieceType.P.value] <= alpha:
            continue
        
        child = SearchPos(node.position)
        child.position.make_move(move)

        score = -qsearch(child, si, -beta, -alpha, depth-1, pv_node)
        
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
    tt.save_tt_entry(tt.TTEntry(node.position.zobrist_hash, best_move.compact(),
                                bound_type, best_value, depth))
    return best_value

def print_moves(moves):
    print(' '.join(map(str,moves)))
