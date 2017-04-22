import pyximport
# pyximport.install()
# from bb import *
import time
from position import Position
from search import Engine
import nn_evaluate
import math
from collections import defaultdict
from evals import BaseEvaluator

baseEvaluator = BaseEvaluator()        
def static_evaluate(position):
    baseEvaluator.set_position(position)
    return baseEvaluator.evaluate()

def parse_epd(epd):
    ops = epd.split(';')
    fen = ops[0].split(' ')
    fen = ' '.join(fen[0:4]) + ' 0 0'

    scores_str, moves_str = ops[3], ops[4]
    scores_str = None
    moves_str = None
    for op in ops[1:]:
        if op.strip().startswith('c8'):
            scores_str = op
        elif op.strip().startswith('c9'):
            moves_str = op
    
    scores = scores_str.split('"')[1]
    scores = [int(s) for s in scores.split(' ')]
    uci_moves = moves_str.split('"')[1]
    uci_moves = uci_moves.split(' ')
    return (fen, uci_moves[0], dict(zip(uci_moves, scores)))

def run_sts_test():
    total_scores = defaultdict(int)
    best_counts = defaultdict(int)
    n = 0
    with open("./tools/STS1-STS15.EPD") as f:
        for epd in f.readlines():
            n += 1
            epd_num = int(math.ceil(n/100))
            fen, best_move, move_scores = parse_epd(epd)

            psn = Position.from_fen(fen)

            engine = Engine()
            engine.init_move_history()
            engine.max_depth = 2
            engine.movetime = 120
            engine.root_position = psn
            engine.info = lambda *args: True
            engine.debug_info = lambda *args: True
            engine.evaluate = nn_evaluate.evaluate
            # engine.evaluate = static_evaluate
            engine.search_stats.node_count = 0
            engine.search_stats.time_start = time.time()

            leaf_val, si = engine.iterative_deepening()
            chosen_move = si[0].pv[0]

            print("\n{0}/1500".format(n))
            print(psn)
            print("chosen", chosen_move.to_uci)
            print("best", best_move)
            print(move_scores)
            score = move_scores.get(chosen_move.to_uci) or 0
            print("score", score)
            total_scores[epd_num] += score
            if chosen_move.to_uci == best_move:
                best_counts[epd_num] += 1

    final_score = sum(total_scores.values())
    best_count = sum(best_counts.values())

    print("Best counts:", best_counts)
    print("Scores:", total_scores)
    print()
    print("Final best counts", best_count)
    print("Final score:", final_score)
    a, b = .359226, 10.402545
    print("Estimated ELO rating:", a * final_score + b)
    return final_score

if __name__ == "__main__":
    now = time.time()
    run_sts_test()
    print(time.time() - now, 's')
