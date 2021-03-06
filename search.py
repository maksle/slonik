from IPython import embed
import time 
import pprint
from collections import namedtuple, Counter as Cntr, defaultdict
from position import *
from constants import *
from material import *
from psqt import *
from evals import *
from operator import itemgetter
from features import ToFeature
import numpy as np
import random
import nn_evaluate
import threading
import itertools
import logging
import logging_config
import sys
import tt

# from nn import model #mlp
from cnnfeat import get_feats #cnn
from nn_evaluate import model

log = logging.getLogger(__name__)

# history moves
History = namedtuple('History', ['move', 'value'])

# counter moves
Counter = namedtuple('Counter', ['move'])


class SearchInfo():
    def __init__(self):
        self.ply = 0
        self.pv = []
        self.train_error = 0
        self.null_move_prune_search = False
        self.skip_early_pruning = False
        self.static_eval = None
        self.singular_search = False
        self.excluded_move = Move(PieceType.NULL)


class SearchStats():
    def __init__(self):
        self.reset()

        # To keep track of search elapse time and for stop search checks
        self.time_start = 0

        # Keep track of checkpoints to throttle the stop search check
        self.checkpoints = 0

        # Move count at which new best move was found. If the move count is low
        # we are sorting the candidate moves well
        self.move_count_pv = Cntr()
        
    def update_ply_stat(self, ply, pv_node):
        if pv_node:
            self.ply['count'] += ply
            self.ply['iter'] += 1
            self.ply['max'] = max(self.ply['max'], ply)

    def reset(self):
        # Table base hits counter for uci output
        self.tb_hits = 0
        # Depth count stats
        self.ply = dict(count=0, iter=0, max=0)
        # Node count for uci output
        self.node_count = 0
        
def depth_to_allowance(depth):
    """convert depth to approximate equivalent allowance"""
    return int(math.ceil(.193006 * math.e ** (1.29829 * depth))) * 2

def allowance_to_depth(allowance):
    """convert allowance to approximate equivalent depth"""
    base = allowance / .193006
    if base <= 0: return 0
    return round(math.log(base) / 1.29829, 1) / 2

def print_moves(moves):
    """Prints the moves to short algebraic notation"""
    print(' '.join(map(str,moves)))


class TimeManagement():
    def __init__(self):
        # time in seconds
        self.wtime = 0
        self.btime = 0
        self.movestogo = None

    def calculate_movetime(self, root_position):
        side = root_position.side_to_move()
        current_move_number = (len(root_position.moves) + 1) // 2
        if side == Side.WHITE:
            time = self.wtime
        else:
            time = self.btime
        moves_to_go = self.movestogo
        if moves_to_go is None:
            moves_to_go = 65 - current_move_number
        if moves_to_go <= 0:
            return .95 * time
        out_of_book_move_number = 0
        moves_out_of_book = min(10, current_move_number - out_of_book_move_number)
        return int(.95 * (time / moves_to_go) * (2 - moves_out_of_book / 10))

baseEvaluator = BaseEvaluator()        
def static_evaluate(position):
    baseEvaluator.set_position(position)
    return baseEvaluator.evaluate()

class Engine(threading.Thread):
    # Engine runs in a thread so we can respond to uci commands while thinking

    def __init__(self, use_static_evaluator=False):
        super(Engine, self).__init__()

        # Whether we are just playing or doing RL backups
        self.training = False
        # The visited states for the tree strap backup 
        self.search_states = []
        
        # root position from which to search
        self.root_position = Position()
        self.next_root_position = None
        # moves available in root position, can be set by uci commands
        self.root_moves = None
        self.next_root_moves = None

        # stack info and for finding principal variations
        self.si = [None] * 64

        # Evaluator (static or nn)
        if use_static_evaluator:
            self.evaluate = static_evaluate
        else:
            self.evaluate = nn_evaluate.evaluate
        
        # ply counts and other stats
        self.search_stats = SearchStats()

        # heuristics
        self.killer_moves = defaultdict(list)
        self.counter_history = [[[None for sq in range(64)] for piece in range(13)] for side in range(2)]
        self.move_history = self.init_move_history()

        # to enable debug statements
        self.debug = True
        
        # printing functions, as for uci output
        self.info = print
        self.debug_info = print

        self.time_management = TimeManagement()
        self.movetime = None
        
        # uci ponder mode
        self.ponder = False
        # uci infinite thinking mode
        self.infinite = False
        # uci max nodes mode
        self.max_nodes = None
        # uci max depth mode can set this
        self.max_depth = None

        # events to respond to uci commands
        self.exec_event = threading.Event()
        self.stop_event = threading.Event()
        self.quit_event = threading.Event()
        self.is_stopped = threading.Event()
        self.is_stopped.set()
    
    def new_game(self, fen="", uci_moves=None):
        """called by uci "position" command to set new root position"""
        if fen == "":
            next_position = Position()
        else:
            next_position = Position.from_fen(fen)
        if uci_moves:
            for uci_move in uci_moves:
                move = next_position.uci_move_to_move(uci_move)
                next_position.make_move(move)
        self.next_root_position = next_position
            
    def uci_root_moves(self, uci_moves=None):
        """Set by the "searchmoves" option of the uci "go" command, which limits
        the moves to search in the root position"""
        if uci_moves is None:
            moves = self.root_position.generate_moves_all(legal=True)
        else:
            moves = [self.root_position.uci_move_to_move(m) for m in uci_moves]
        self.next_root_moves = [[move, 0] for move in moves]
    
    def run(self):
        """override of threading.Thread method. This kicks off the thread"""
        while self.exec_event.wait():
            if self.quit_event.is_set():
                sys.exit()
            try:
                self.iterative_deepening()
            except:
                log.exception("search error")
                raise
            self.stop_event.clear()
            self.exec_event.clear()
            self.is_stopped.set()
            if self.quit_event.is_set():
                sys.exit()
        
    def stop(self):
        self.stop_event.set()
        
    def quit(self):
        self.quit_event.set()
        self.stop_event.set()
        self.exec_event.set()
        
    def go(self):
        """respond to uci "go" command"""
        self.stop_event.set()
        self.is_stopped.wait()
        self.is_stopped.clear()

        # reset stats
        self.search_stats.node_count = 0
        self.search_stats.time_start = time.time()

        # initialize position
        if self.next_root_position:
            self.root_position, self.next_root_position = self.next_root_position, None
            self.root_moves, self.next_root_moves = self.next_root_moves, None
            self.init_move_history()
        
        # time management
        if (not (self.infinite or self.ponder or self.movetime or self.max_depth)):
            self.movetime = self.time_management.calculate_movetime(self.root_position)
        if self.max_depth is None:
            self.max_depth = 64

        self.stop_event.clear()
        self.exec_event.set()
        
    def checkpoint(self):
        """Check if we should stop searching"""
        # max_depth is taken care of in iterative_deepening()
        if (self.max_nodes and self.search_stats.node_count >= self.max_nodes) \
           or (self.movetime is not None and (time.time() - self.search_stats.time_start) >= self.movetime):
            self.stop_event.set()
        
    def init_move_history(self):
        return [[[
            History(Move(piece, to_sq=1<<sq), psqt_value_sq(piece, 1<<sq, side))
            for sq in range(64)] for piece in range(13)] for side in range(2)
        ]
    
    def rotate_killers(self):
        d = defaultdict(list)
        for ply, v in self.killer_moves.items():
            d[ply-1] = self.killer_moves[ply]
        if -1 in d:
            del d[-1]
        self.killer_moves = d
                
    # History heuristic
    def update_history(self, side, move, value):
        """Maintain a score for piece/square composite key as a rudimentary heuristic"""
        entry = self.move_history[side][move.piece_type][bit_position(move.to_sq)]
        learn_rate = .3
        discount = .9
        if entry:
            adjusted_val = entry.value + learn_rate * (discount * value - entry.value)
            self.move_history[side][move.piece_type][bit_position(move.to_sq)] = History(move, adjusted_val)

    def lookup_history(self, side, move):
        if move and move.piece_type is not PieceType.NULL:
            return self.move_history[side][move.piece_type][bit_position(move.to_sq)]

    # Counters heuristic
    def update_counter(self, side, move, counter):
        """Maintain record of refuted/refuter move pairs"""
        entry = self.counter_history[side][move.piece_type][bit_position(move.to_sq)]
        self.counter_history[side][move.piece_type][bit_position(move.to_sq)] = Counter(counter)

    def lookup_counter(self, side, move):
        if move and move.piece_type is not PieceType.NULL:
            return self.counter_history[side][move.piece_type][bit_position(move.to_sq)]

    # Killer heuristic
    def add_killer(self, ply, move):
        """Maintain list of moves found to be strong at the given depth"""
        if move in self.killer_moves[ply]:
            return
        if len(self.killer_moves[ply]) == 3:
            self.killer_moves[ply].pop()
        self.killer_moves[ply].append(move)

    # Root move sorting
    def update_root_score(self, move, score):
        for rmove in self.root_moves:
            if rmove[0] == move:
                rmove[1] = score
                break
        
    # Search
    def iterative_deepening(self):
        """Search deeper each iteration until stop condition. If fail high or
        fail low, widen the aspiration window and search again."""
        
        alpha = LOW_BOUND
        beta = HIGH_BOUND

        depth = allowance_to_depth(35)

        val = self.evaluate(self.root_position)

        pv = []
        
        # move ordering of root moves, case when uci doesn't set the "searchmoves"
        if self.root_moves is None:
            self.root_moves = [[move, 0] for move in self.root_position.generate_moves_all(legal=True)]
        
        # log.debug("Side to move: %s", self.root_position.side_to_move())
        
        # log.debug("max_depth %s", self.max_depth)
        while not self.stop_event.is_set() \
              and depth <= self.max_depth \
              and (not self.max_nodes or self.search_stats.node_count < self.max_nodes) \
              and (self.movetime is None or (time.time() - self.search_stats.time_start) < self.movetime):
            # log.debug("searching depth %s, max_depth %s", depth, self.max_depth)
            depth += 1
            allowance = depth_to_allowance(depth)
            finished = False
            fail_factor = 18
            self.search_stats.reset()
            
            alpha, beta = max(LOW_BOUND, val - fail_factor), min(HIGH_BOUND, val + fail_factor)
            bound = ""
            
            while not finished:

                if self.stop_event.is_set(): break
                val = self.search(self.root_position, 0, alpha, beta, allowance, True, False)
                if self.stop_event.is_set(): break
                
                if val <= alpha:
                    self.debug_info("faillow", "a", alpha, "b", beta, "val", val, "factor", fail_factor, "allowance", allowance)
                if val >= beta:
                    self.debug_info("failhigh", "a", alpha, "b", beta, "val", val, "factor", fail_factor, "allowance", allowance)
                    
                # log.debug("BEFORE: val: %s, alpha: %s, beta: %s", val, alpha, beta)
                    
                if LOW_BOUND < val <= alpha:
                    bound = " upperbound"
                    alpha = max(LOW_BOUND, val - fail_factor)
                    # beta = (alpha + beta) // 2
                    # fail_factor += fail_factor // 3 + 6
                    fail_factor *= 3
                elif HIGH_BOUND > val >= beta:
                    bound = " lowerbound"
                    beta = min(HIGH_BOUND, val + fail_factor)
                    # alpha = (alpha + beta) / 2
                    # fail_factor += fail_factor // 3 + 6
                    fail_factor *= 3
                    pv = self.si[0].pv[:]
                else:
                    bound = ""
                    finished = True
                    pv = self.si[0].pv[:]

                # log.debug("AFTER: val: %s, alpha: %s, beta: %s, ply: %s, finished: %s",
                #           val, alpha, beta,
                #           int(self.search_stats.ply["count"] / (self.search_stats.ply["iter"] or 1)),
                #           finished)
                    
                pv = self.si[0].pv[:]
                # log.debug(pv)
                # pv2 = self.find_pv(self.root_position)
                
                if bound == "":
                    elapsed = time.time() - self.search_stats.time_start
                    self.info("info depth", int(self.search_stats.ply["count"] / (self.search_stats.ply["iter"] or 1)),
                              "seldepth", int(self.search_stats.ply["max"]),
                              "score cp", str(int(val * 100 / EG_PIECES[Pt.P])) + bound,
                              "nodes", self.search_stats.node_count,
                              "nps", int(self.search_stats.node_count / (elapsed or 1)),
                              "tbhits", self.search_stats.tb_hits,
                              "time", int(elapsed),
                              "pv", " ".join([m.to_uci for m in pv]))

                # log.info("allowance, avg pv ply, max pv ply: %s, %s, %s",
                #          allowance,
                #          int(self.search_stats.ply["count"] / (self.search_stats.ply["iter"] or 1)),
                #          int(self.search_stats.ply["max"]))

                # self.debug_info("most common move_counts:", sum(self.search_stats.move_count_pv.values()), self.search_stats.move_count_pv.most_common(5))
                # self.info("string", "pv2", " ".join([m.to_uci for m in pv2]))

                # log.info("allowance, nodes, elapsed: %s %s %s", allowance, self.search_stats.node_count, elapsed)
        
        # some stop condition but stop event didn't come yet, we have to wait
        if not self.stop_event.is_set() and (self.ponder or self.infinite):
            self.stop_event.wait()
            
        if len(pv) == 0:
            pv = self.find_pv(self.root_position)
            
        if len(pv) > 1:
            self.info("bestmove", pv[0].to_uci, "ponder", pv[1].to_uci)
        elif len(pv) > 0:
            self.info("bestmove", pv[0].to_uci)
        else:
            log.debug("no pv, no bestmove, exiting iterative_deepening")
        
        return val, self.si
    
    def search(self, node, ply, a, b, allowance, pv_node, cut_node):
        """Search for best move in position `node` within alpha and beta window."""
        
        a_orig = a
        is_root = pv_node and ply == 0
        
        self.search_stats.node_count += 1
        
        if self.search_stats.checkpoints > 500: # throttle
           self.search_stats.checkpoints = 0               
           self.checkpoint()
        
        if self.stop_event.is_set():
            return STOP_VALUE

        assert(pv_node or a == b-1)
        
        si = self.si
        si[ply] = si[ply] or SearchInfo()
        si[ply+1] = si[ply+1] or SearchInfo()
        if not is_root:
            si[ply].pv.clear()
        si[ply+1].pv.clear()

        if arbiter_draw(node):
            return DRAW_VALUE
        
        pos_key = node.zobrist_hash ^ si[ply].excluded_move.compact()
        
        found = False
        found, tt_ind, tt_entry = tt.get_tt_index(pos_key)
        if not pv_node and found and tt_entry.depth >= allowance:
            if (tt_entry.bound_type == tt.BoundType.LO_BOUND and tt_entry.value >= b) \
               or (tt_entry.bound_type == tt.BoundType.HI_BOUND and tt_entry.value < a) \
               or tt_entry.bound_type == tt.BoundType.EXACT:
                self.search_stats.tb_hits += 1
                return tt_entry.value
        
        # if not self.training and found and tt_entry.static_eval is not None:
        #     si[ply].static_eval = static_eval = tt_entry.static_eval
        # else:
        #     if node.last_move() == Move(PieceType.NULL):
        #         si[ply].static_eval = static_eval = -si[ply-1].static_eval + 40
        #     else:
        #         si[ply].static_eval = static_eval = self.evaluate(node)
        #     tt.save_tt_entry(tt.TTEntry(pos_key, 0, tt.BoundType.NONE, 0, 0, static_eval))

        # if node.last_move() != Move(PieceType.NULL) and (abs(static_eval) - abs(self.evaluate(static_eval)) > .0008):
        #     print("in search")
        #     embed()
            
        in_check = node.in_check()

        if allowance < 1 and in_check:
            allowance = 1
        
        if allowance < 1:
            score = self.qsearch(node, ply, 0, a, b, pv_node, in_check)
            return score
        
        # if not pv_node and not in_check:
        #     if not si[ply].skip_early_pruning:
        #         # futility prune of parent
        #         if not pv_node \
        #            and allowance_to_depth(allowance) < 6 \
        #            and static_eval - allowance_to_depth(allowance) * 150 >= b:
        #             return static_eval

        #         # null move pruning.. if pass move and we're still failing high, dont bother searching further
        #         # if not pv_node and not si[ply].null_move_prune_search \
        #         #    and found and tt_entry.value >= b:
        #         #     si[ply+1].null_move_prune_search = True
        #         #     node.make_null_move()
        #         #     val = -self.search(node, ply+1, -b, -b+1, int(allowance * .3), False, False)
        #         #     node.undo_null_move()
        #         #     si[ply+1].null_move_prune_search = False

        #         #     if val >= b:
        #         #         return val

        #     # internal iterative deepening to improve move order when there's no pv
        #     if not found and allowance_to_depth(allowance) >= 4 \
        #        and (pv_node or static_eval + MG_PIECES[PieceType.P] >= b):
        #         si[ply+1].skip_early_pruning = True
        #         val = self.search(node, ply, a, b, int(allowance * .75), pv_node, cut_node)
        #         si[ply+1].skip_early_pruning = False
        #         found, tt_ind, tt_entry = tt.get_tt_index(node.zobrist_hash)
        
        # singular_extension_eligible = False
        # if not is_root and allowance_to_depth(allowance) >= 2.5 \
        #    and not si[ply].singular_search \
        #    and found \
        #    and tt_entry.move != 0 \
        #    and tt_entry.bound_type != tt.BoundType.HI_BOUND \
        #    and tt_entry.depth >= allowance * .7:
        #     singular_extension_eligible = True

        # improving = (ply < 2) or (si[ply-2].static_eval is None) or (si[ply].static_eval >= si[ply-2].static_eval)

        best_move = Move(PieceType.NULL)
        best_move_is_capture = False
        best_val = LOW_BOUND

        move_count = 0
        best_val_move_count = 0
            
        if is_root and any(map(lambda x: x[1] != 0, self.root_moves)):
            moves = map(itemgetter(0), sorted(self.root_moves, key=itemgetter(1), reverse=True))
        elif is_root:
            moves = map(itemgetter(0), self.root_moves)
            moves = self.sort_moves(moves, node, si, ply)
            self.root_moves = [[move, ind * .001] for (ind, move) in enumerate(moves[::-1])]
        else:
            if in_check: pseudo_moves = node.generate_moves_in_check()
            else: pseudo_moves = node.generate_moves_all()
            moves = self.sort_moves(pseudo_moves, node, si, ply)
            
        # counter = None
        # if len(node.moves):
        #     counter = self.lookup_counter(node.side_to_move() ^ 1, node.moves[-1])
        
        #
        # Move iteration
        for move in moves:
   
            val = 0
            
            # if not improving:
            #     move.prob = min(move.prob * .9, 1)
            
            if si[ply].excluded_move == move:
                continue

            if not node.is_legal(move, in_check):
                continue

            if is_root and len(si[0].pv) == 0:
                si[0].pv = [move]
                
            child = Position(node)
            child.make_move(move)
            move_count += 1

            gives_check = child.in_check()
            if gives_check:
                move.move_type |= MoveType.check

            # is_capture = move.to_sq & node.occupied[child.side_to_move()]
            # see_score = move.see_score or eval_see(node, move)

            # extending = False

            # if gives_check and move.prob > .01 and see_score > 0:
            #     extending = True
            #     move.prob = 1

            # if not extending and pv_node and len(node.moves) > 0 and node.moves[-1] != Move(PieceType.NULL):
            #     # recapture extension
            #     if node.moves[-1].move_type & MoveType.capture and node.moves[-1].to_sq == move.to_sq:
            #        extending = True
            #        move.prob = min(move.prob * 1.5, 1)

            # singular extension logic pretty much as in stockfish
            # if not extending and singular_extension_eligible and move == Move.move_uncompacted(tt_entry.move):
            #     # print("Trying Singular:", ' '.join(map(str,node.moves)), move)
            #     rbeta = int(tt_entry.value - (2 * allowance_to_depth(allowance)))
            #     si[ply].excluded_move = move
            #     si[ply].singular_search = True
            #     si[ply].skip_early_pruning = True
            #     val = self.search(node, ply, rbeta-1, rbeta, int(allowance * move.prob * .75), False, cut_node)
            #     si[ply].skip_early_pruning = False
            #     si[ply].singular_search = False
            #     si[ply].excluded_move = Move(PieceType.NULL)
                # print("val", val, "rbeta", rbeta)

                # if val < rbeta:
                #     # print("extending", node.moves, move)
                #     extending = True
                #     move.prob = 1
            
            # if not is_root and not is_capture and not gives_check:
            #     next_depth = allowance_to_depth(allowance * move.prob)
            #     if next_depth <= 5 and not in_check \
            #        and static_eval + 1.25 * MG_PIECES[PieceType.P] + next_depth * MG_PIECES[PieceType.P] <= a:
            #         continue
            
            # TODO?: if non rootNode and we're losing, only look at checks/big captures >= alpha 

            # # Probabilistic version of LMR
            # # .. zero window search reduced 
            # do_full_zw = False
            # zw_allowance = allowance * move.prob 
            # if not pv_node:
            #     if allowance_to_depth(allowance) >= 3 and not is_capture:
            #         r = min(zw_allowance * .25, depth_to_allowance(1))
            #         if cut_node:
            #             r += depth_to_allowance(2)
            #         else:
            #             child.make_null_move()
            #             undo_see = eval_see(child, Move(move.piece_type, move.to_sq, move.from_sq))
            #             child.undo_null_move()
            #             if undo_see < 0:
            #                 # reduce reduction if escaping capture
            #                 r -= depth_to_allowance(2)
            #             else:
            #                 # reduce reduction if making a threat
            #                 ep_default_value = (0, 0, 0, 0)
            #                 victim_before, *rest = next_en_prise(node, child.side_to_move())
            #                 victim_after, *rest = next_en_prise(child, child.side_to_move())
            #                 if victim_after > victim_before:
            #                     r -= depth_to_allowance(2)

            #         hist = self.lookup_history(node.side_to_move(), move)
            #         # log.debug("hist %s %s", move, hist)
            #         if hist and hist.value > .5:
            #             r -= depth_to_allowance(1)
            #         elif hist and hist.value < 0:
            #             r += depth_to_allowance(1)

            #         if r < 0: r = 0

            #         val = -self.search(child, ply+1, -(a+1), -a, int(zw_allowance - r), False, True)
            #         do_full_zw = val > a and r != 0
            #     else:
            #         do_full_zw = not (pv_node and move.prob >= .035)

            #     # .. zero window full allotment search
            #     if do_full_zw:
            #         val = -self.search(child, ply+1, -(a+1), -a, int(zw_allowance), True, not cut_node)
                
            # # .. full window full allotment search
            # # otherwise we let the fail highs cause parent to fail low and try different move
            # if (pv_node or (do_full_zw and a < val <= b)):
            #     val = -self.search(child, ply+1, -b, -a, int(allowance * move.prob), True, False)
            
            if pv_node and move_count > 1 and allowance_to_depth(allowance) > 2:
                val = -self.search(child, ply+1, -(a+1), -a, int(allowance * move.prob), False, True)
                if a < val < b:
                    val = -self.search(child, ply+1, -b, -a, int(allowance * move.prob), True, False)
            else:
                val = -self.search(child, ply+1, -b, -a, int(allowance * move.prob), True, False)
                
            if is_root:
                self.update_root_score(move, val)
                
            if val > best_val:
                prev_best = best_move
                best_val, best_move = val, move
                best_move_is_capture = is_capture
                best_val_move_count = move_count
                # if pv_node and val > a:
                # if pv_node:
                si[ply].pv = [move] + si[ply+1].pv
                
            if self.stop_event.is_set():
                return STOP_VALUE
                
            a = max(a, val)

            if a >= b:
                # captures already get searched before killers
                # if move == best_move and not is_capture: 
                #     self.add_killer(ply, best_move)
                break
        
        self.search_stats.move_count_pv[best_val_move_count] += 1
            
        prior_move = node.last_move()
        if move_count == 0:
            if si[ply].excluded_move != Move(PieceType.NULL): 
                best_val = a
            # mate or statemate
            elif in_check:
                best_val = MATE_VALUE + ply
            else:
                best_val = DRAW_VALUE
        # elif best_move != Move(PieceType.NULL):
            # if best_val > a_orig and not best_move_is_capture:
            #     bonus = int(allowance_to_depth(allowance))
            #     self.update_history(node.side_to_move(), best_move, bonus)
            #     if prior_move and not prior_move.is_null_move():
            #         self.update_counter(node.side_to_move() ^ 1, prior_move, best_move)
            #         # penalize prior quiet move that allowed this good move
            #         if len(node.moves) > 1 and not prior_move.is_null_move and not prior_move.move_type & MoveType.capture:
            #             # print("penalizing", node.side_to_move() ^ 1, "for move", prior_move, "with", -(bonus + 4))
            #             self.update_history(node.side_to_move() ^ 1, prior_move, -(bonus + 4))
        # elif best_val <= a_orig and allowance_to_depth(allowance) >= 2.5 and not best_move_is_capture:
        #     # reward the quiet move that caused this node to fail low
        #     bonus = int(allowance_to_depth(allowance))
        #     if len(node.moves) > 1 and not prior_move.is_null_move() and not prior_move.move_type & MoveType.capture:
        #         # print("rewarding", node.side_to_move() ^ 1, "for move", prior_move, "with", bonus)
        #         self.update_history(node.side_to_move() ^ 1, prior_move, bonus)
        
        if best_val <= a_orig: bound_type = tt.BoundType.HI_BOUND
        elif best_val >= b: bound_type = tt.BoundType.LO_BOUND
        else: bound_type = tt.BoundType.EXACT
        
        static_eval = None
        if self.training and node.last_move() != Move(PieceType.NULL) and allowance >= 1:
            static_eval = self.evaluate(node)
            self.search_states.append((node.fen(), best_val, bound_type, static_eval))

        tt.save_tt_entry(tt.TTEntry(pos_key, best_move.compact(),
                                    bound_type, best_val, allowance, static_eval))
        
        if is_root: assert(len(si[0].pv) > 0)
        return best_val

    def qsearch(self, node, ply, qsply, alpha, beta, pv_node, in_check):
        """Find best resulting quiet position within the alpha-beta window and
        return the evaluation."""

        self.search_stats.node_count += 1
        
        if self.search_stats.checkpoints > 500: # throttle
           self.search_stats.checkpoints = 0               
           self.checkpoint()

        if self.stop_event.is_set():
            return STOP_VALUE

        assert(pv_node or alpha == beta-1)
        
        si = self.si
        si[ply] = si[ply] or SearchInfo()
        si[ply+1] = si[ply+1] or SearchInfo()
        si[ply].pv.clear()
        si[ply+1].pv.clear()

        if arbiter_draw(node):
            return DRAW_VALUE
        
        tt_hit = False
        a_orig = alpha

        # if ' '.join(map(str, node.moves)) == 'e2-e4 e7-e6 Qd1-f3':
        #     print("debug")

        tt_hit, tt_ind, tt_entry = tt.get_tt_index(node.zobrist_hash)
        if tt_hit:
            self.search_stats.tb_hits += 1

        if not pv_node:
            if tt_hit and tt_entry.depth >= 0 and tt_entry.bound_type != tt.BoundType.NONE:
                if tt_entry.bound_type == tt.BoundType.EXACT:
                    self.search_stats.update_ply_stat(ply, pv_node)
                    # log.debug("a %s b %s moves %s exact tt_entry.value %s", alpha, beta, node.moves, tt_entry.value)
                    return tt_entry.value

                if tt_entry.bound_type == tt.BoundType.LO_BOUND and tt_entry.value >= beta:
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.bound_type == tt.BoundType.HI_BOUND and tt_entry.value < alpha:
                    beta = min(beta, tt_entry.value)

                if alpha >= beta:
                    self.search_stats.update_ply_stat(ply, pv_node)
                    # log.debug("a %s b %s moves %s lowbound tt_entry.value %s", alpha, beta, node.moves, tt_entry.value)
                    return tt_entry.value
   
        # if in_check and qsply > 1:
        #     return self.search(node, ply, alpha, beta, 1, pv_node, False)
            
        static_eval = None
        if tt_hit and not self.training:
            static_eval = tt_entry.static_eval
        if static_eval is None:
            static_eval = self.evaluate(node)
        
        if not in_check:
            if static_eval >= beta:
                # We can't stand pat if in check, because standing pat assumes that
                # there is at least some quiet move, if not violent move, that is at
                # least as good as alpha. We can't assume that if we can't deal with
                # the check.

                # "stand pat"
                if not tt_hit or tt_entry.bound_type == tt.BoundType.NONE:
                    # d = QS_CHECK_DEPTH if qsply == 0 else QS_DEPTH
                    d = QS_DEPTH
                    tt.save_tt_entry(tt.TTEntry(node.zobrist_hash,
                                                Move(PieceType.NULL).compact(),
                                                tt.BoundType.LO_BOUND, static_eval, d, static_eval))
                self.search_stats.update_ply_stat(ply, pv_node)
                # if pv_node:
                #     print("qs stand pat", node.moves, best_value, ">=", beta)
                # log.debug("a %s b %s moves %s standpat static_eval %s", alpha, beta, node.moves, static_eval)
                return static_eval

            if pv_node and static_eval > alpha:
                alpha = static_eval
        
        best_move = Move(PieceType.NULL)
        move_count = 0

        if in_check:
            pseudo_moves = node.generate_moves_in_check()
        else:
            pseudo_moves = node.generate_moves_violent(do_checks= qsply == 0)
        moves = self.sort_moves(pseudo_moves, node, si, ply)
        # log.debug("so far: %s violent: %s", node.moves, moves)

        for move in moves:
            # if ' '.join(map(str,node.moves)) == "Ng4-e5 Nf3-e5 Nc6-e5 Bc4-f7 Ke8-e7" and str(move) == "Rf1-e1":
            #     print("debug")

            score = 0
            
            if not node.is_legal(move, in_check):
                continue

            child = Position(node)
            child.make_move(move)
            move_count += 1

            gives_check = child.in_check()
            if gives_check:
                move.move_type |= MoveType.check

            # is_capture = move.to_sq & node.occupied[node.side_to_move() ^ 1]

            # if not in_check and not gives_check:
            #     # Futility pruning
            #     # .. try to avoid calling eval_see
            #     pt_captured = node.squares[bit_position(move.to_sq)]
            #     if static_eval + MG_PIECES[PieceType.base_type(pt_captured)] + MG_PIECES[PieceType.P] <= alpha:
            #         continue
            #     see_score = move.see_score if move.see_score is not None else eval_see(node, move)
            #     if static_eval + see_score + MG_PIECES[PieceType.P] <= alpha \
            #        and see_score < 0:
            #         continue

            # if not in_check or not is_capture:
            #     see_score = move.see_score if move.see_score is not None else eval_see(node, move)
            #     if see_score < 0:
            #         continue

            score = -self.qsearch(child, ply+1, qsply+1, -beta, -alpha, pv_node, gives_check)
            
            if score > alpha:
                alpha = score
                si[ply].pv = [move] + si[ply+1].pv

            if alpha >= beta:
                d = QS_CHECK_DEPTH if qsply == 0 else QS_DEPTH
                tt.save_tt_entry(tt.TTEntry(node.zobrist_hash, move.compact(),
                                            tt.BoundType.LO_BOUND, alpha, d, static_eval))
                self.search_stats.update_ply_stat(ply, pv_node)
                return alpha
    
        if in_check and move_count == 0:
            return MATE_VALUE + ply
            
        if pv_node and alpha > a_orig: bound_type = tt.BoundType.EXACT
        else: bound_type = tt.BoundType.HI_BOUND
        d = QS_CHECK_DEPTH if qsply == 0 else QS_DEPTH
        tt.save_tt_entry(tt.TTEntry(node.zobrist_hash, best_move.compact(),
                                    bound_type, alpha, d, static_eval))
        self.search_stats.update_ply_stat(ply, pv_node)

        # log.debug("a %s b %s moves %s best_value (in bounds) %s", alpha, beta, node.moves, alpha)
        return alpha

    def find_pv(self, root_pos):
        """ Search the transposition table for the principal variation."""
        moves = []
        pos = Position(root_pos)

        def find_next_move():
            found, tt_ind, tt_entry = tt.get_tt_index(pos.zobrist_hash)
            if found:
                move = Move.move_uncompacted(tt_entry.move)
                if not move.is_null_move() and tt_entry.bound_type in [tt.BoundType.EXACT, tt.BoundType.LO_BOUND]:
                    return True, move
                else:
                    return False, move
            return False, 0

        found, move = find_next_move()
        while found and len(moves) < self.max_depth:
            moves.append(move)
            pos.make_move(move)
            found, move = find_next_move()

        return moves
    
    def sort_moves(self, moves, position, si, ply, is_root=False):
        """Sort the moves to optimize the alpha-beta search"""
        # toFeature = ToFeature()
        moves = list(moves)
        if len(moves) == 0:
            return moves
        feats = []
        for move in moves:
            pos_ = Position(position)
            pos_.make_move(move)
            # toFeature.set_position(pos_)
            # feats.append(toFeature.ann_features())
            feats.append(get_feats(pos_))
        feats = np.stack(feats)
        # if len(moves) >= 20:
        #     predictions = model.target.predict(feats)
        # else:
        #     predictions = model.actor.predict(feats)
        predictions = model.predict(feats, batch_size=len(feats))
        predictions = predictions * -1 if position.side_to_move() == Side.BLACK else predictions
        moves_preds_sorted = list(sorted(zip(moves, predictions.tolist()), key=lambda x: x[1], reverse=True))
        preds_sorted = np.array([pred for (move, pred) in moves_preds_sorted])
        if len(preds_sorted) == 0:
            embed()
        e_x = np.exp(preds_sorted - preds_sorted.max()) ** .5 # soften the temperature
        probs = e_x / e_x.sum()
        assert(np.allclose(sum(probs), 1))
        sorted_moves = [m[0] for m in moves_preds_sorted]
        for i, move in enumerate(sorted_moves):
            move.prob = probs[i]
        return sorted_moves
        
        # from_pv = [] 
        # captures = []
        # killers = []
        # counters = []
        # other_moves = []
        # checks = []

        # side = position.side_to_move()
        # us, them = side, side ^ 1

        # other = position.occupied[them]
        # counter = self.lookup_counter(them, position.last_move())

        # # ep_us_before = next_en_prise(position, us)
        # # ep_them_before = next_en_prise(position, them)

        # def sort_crit(move):
        #     entry = self.lookup_history(us, move)
        #     # see_val = eval_see(position, move)
        #     hist_val = entry.value if entry else 0
        #     # return (see_val, hist_val)
        #     return (0, hist_val)

        # pv_moves = self.find_pv(position)
        # found, tt_ind, tt_entry = tt.get_tt_index(position.zobrist_hash)
        # from_tt = []
        # for move in moves:
        #     if move in pv_moves:
        #         from_pv.append(move)
        #     elif found and tt_entry.move != 0:
        #         from_tt.append(move)
        #     elif is_capture(move.to_sq, other):
        #         captures.append(move)
        #     elif move.move_type == MoveType.check:
        #         checks.append(move)
        #     elif counter and counter.move == move:
        #         counters.append(move)
        #     elif move in self.killer_moves[ply]:
        #         killers.append(move)
        #     else:
        #         other_moves.append(move)

        # # keep same order as pv
        # from_pv_final = []
        # for move in pv_moves:
        #     if move in from_pv:
        #         from_pv_final.append(move)
        # from_pv = from_pv_final
        
        # # checks = sorted(checks, key=sort_crit, reverse=True)

        # other_moves.sort(key=lambda m: sort_crit(m), reverse=True)
        
        # # captures_see = map(lambda c: (sort_crit(c), c), captures)
        # # sorted_cap_see = sorted(captures_see, key=itemgetter(0), reverse=True)
        # # cap_see_gt0 = []
        # # cap_see_lt0 = []
        # # cap_see_eq0 = []
        # # for cs in sorted_cap_see:
        # #     see, hist = cs[0]
        # #     move = cs[1]
        # #     if see > 0:
        # #         move.prob = 2.5
        # #         cap_see_gt0.append(move)
        # #     elif see == 0:
        # #         move.prob = 1.5
        # #         cap_see_eq0.append(move)
        # #     else:
        # #         move.prob = .75
        # #         cap_see_lt0.append(move)
        
        # for move in from_pv: move.prob = 3
        # for move in from_tt: move.prob = 2.8
        # for move in captures: move.prob = 2.5
        # for move in checks: move.prob = 2
        # for move in counters: move.prob = 2
        # for (ind, move) in enumerate(killers):
        #     move.prob = 1.5 + (ind * .1)
        # for move in other_moves: move.prob = 1
        
        # # result = list(itertools.chain(from_pv, from_tt, cap_see_gt0, checks, counters, killers, cap_see_eq0, other_moves, cap_see_lt0))
        # result = list(itertools.chain(from_pv, from_tt, captures, checks, counters, killers, other_moves))

        # prob_sum = 0
        # for move in result:
        #     prob_sum += move.prob
        # for move in result:
        #     move.prob /= prob_sum
        
        # # result = sorted(result, key=lambda m: m.prob, reverse=True)
        
        # # log.info([move.prob for move in result])
            
        # return result
    
def perft(pos, depth, is_root):
    cnt = nodes = 0
    leaf = depth == 2
    moves = list(pos.generate_moves_all(legal=True))
    for move in moves:
        if is_root and depth <= 1:
            cnt = 1
            nodes += 1
        else:
            child = Position(pos)
            child.make_move(move)
            if leaf:
                cnt = len(list(child.generate_moves_all(legal=True)))
            else:
                cnt = perft(child, depth - 1, False)
            nodes += cnt
        if is_root:
            print(move, cnt)
    return nodes
