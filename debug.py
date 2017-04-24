from search import *
from constants import *
from bb import *
import cProfile
import pstats
import random


pos = Position.from_fen("r2qr1k1/pp2bppp/2bp4/7n/3QP3/2N1BN2/PP3PPP/R3R1K1 w - - 0 1")
def goprofile():
    now = time.time()
    engine = Engine()
    engine.debug = True
    engine.infinite = False
    engine.max_depth = 64
    engine.root_position = pos
    engine.init_move_history()
    engine.search_stats.time_start = time.time()
    engine.iterative_deepening()

# goprofile()

pos = Position.from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
for i in range(1, 6):
    now = time.time()
    print(i, perft(pos, i, True), end=' ')
    print(time.time() - now,'s')

# cProfile.run("goprofile()", filename="outprofile")
# pstats.Stats("outprofile").strip_dirs().sort_stats("time").print_stats(15)
# print()
# pstats.Stats("outprofile").strip_dirs().sort_stats("cumulative").print_stats(15)

# pos = Position.from_fen("r3kbnr/ppp2ppp/2n5/8/2BP2b1/5N2/PP2KPPP/RNB4R w kq - 1 9")
# from evals import BaseEvaluator
# from td import nn_evaluator
# # print(BaseEvaluator(pos).evaluate())
# print(nn_evaluator(pos))

# import time 
# now = time.time()
# val, si = iterative_deepening(3, pos)
# # val = search(SearchPos(pos), [None] * 64, 0, -10000000, 10000000, 1, .001, True)
# # print("node count", node_count)
# print(val / 200)
# then = time.time()
# print(then-now, 's')

# import time
# now = time.time()
# perft(Position(Position.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0")), 3, True)
# then = time.time()
# print(then-now, end='s')

# python3 -m cProfile -o profile5 play.py
# pstats.Stats('profile5').strip_dirs().sort_stats('cumulative').print_stats(15)
