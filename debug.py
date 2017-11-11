from search import *
from constants import *
from bb import *
import cProfile
import pstats
import random
import nn_evaluate


pos = Position.from_fen("k7/8/8/8/P5r1/1P5p/4rr1P/7K b - - 0 1")
def goprofile():
    now = time.time()
    engine = Engine()
    engine.debug = True
    engine.infinite = False
    engine.max_depth = 4
    engine.root_position = pos
    engine.evaluate = nn_evaluate.evaluate
    engine.init_move_history()
    engine.search_stats.time_start = time.time()
    val, si = engine.iterative_deepening()
    print("search res", val)
    print("pv", si[0].pv)
    
    leaf = Position(pos)
    for move in si[0].pv:
        leaf.make_move(move)
    print("eval val", nn_evaluate.evaluate(leaf))
    
# goprofile()

pos = Position.from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
pos = Position.from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0")
pos = Position()
for i in range(5, 6):
    now = time.time()
    nodes = perft(pos, i, True)
    after = time.time()
    print(i, nodes, end=' ')
    print(after - now,'s')
    print("nps", nodes / (after - now))

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
# nodes = perft(Position(Position.from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0")), 5, True)
# print(4, nodes)
# then = time.time()
# print(then-now, end='s')

# f = Position.from_fen("1r2k2r/p1ppqNb1/bn2pnp1/3P4/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQk - 1 2")
# in_check = f.in_check(Move(Pt.N, F7, D6), for_side=Side.B)
# print(in_check)

# f = Position.from_fen("3r2k1/1R3pp1/p3pb1p/P7/8/3b1N1P/1P1BrPP1/R2K4 b - - 16 34")
# print(f.gives_check(Move(Pt.B_ROOK,E2,E1)))


# python3 -m cProfile -o profile5 play.py
# pstats.Stats('profile5').strip_dirs().sort_stats('cumulative').print_stats(15)
