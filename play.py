from search import *
from constants import *
from bb import *
import cProfile
import pstats

# root_pos = Position()
# root = SearchPos(root_pos)
# val, final, child = negamax(root, -100000, 100000, 3, 1)
# val = -val

def play(position):
    root = position or Position()
    
    human_side = input("You play w or b? [w/b]")
    # while True:
    #     try:
    #         human_side = input("You play w or b? [w/b]")
    #         if human_side in ["w", "b"]:
    #             break
    #     except:
    #         pass

    si = [None] * 64
    depth = 4
    
    if (human_side == "w" and root.side_to_move() == Side.BLACK) or \
       (human_side == "b" and root.side_to_move() == Side.WHITE):
        val, si = iterative_deepening(depth, root, si)
        # print("==== Computer move: ==========")
        # print(chosen)
        # print("==============================")
        move = si[0].pv[0]
        root.make_move(move)
        make_move(move)
        
    while not root.is_mate():
        user_move = get_user_move(root.side_to_move())
        # while True:
        #     try:
        #         user_move = get_user_move()
        #     except:
        #         pass
        print("==== Human move: ==============")
        print(user_move)
        print("===============================")
        root.make_move(user_move)
        make_move(user_move)
        
        val, si = iterative_deepening(depth, root, si)
        # print("==== Computer move: ==========")
        # print("==============================")
        move = si[0].pv[0]
        root.make_move(move)
        make_move(move)
        
    print("Mate?:", child.is_mate())
    print("Gave over")
        
def get_user_move(side):
    move = input("Enter move: <piece_type> <from_sq> <to_sq>")
    inputs = move.split()
    if len(inputs) == 3:
        piece_letter, from_sq, to_sq = inputs
    else:
        piece_letter = ""
        from_sq, to_sq = inputs
    # print(piece_type, from_sq, to_sq)
    base_type = PieceType.base_type(HUMAN_PIECE_INV[str.upper(piece_letter)])
    piece_type = PieceType.piece(base_type, side)
    print("Piece type is", piece_type)
    from_sq = HUMAN_BOARD[from_sq]
    to_sq = HUMAN_BOARD[to_sq]
    move = Move(piece_type, from_sq, to_sq)
    return move

# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")
# pos = Position.from_fen("r1bqkb1r/ppp2ppp/3p1n2/4n3/8/2N1P1Q1/PPP2PPP/R1B1KBNR b KQkq - 1 6")
# print(pos.is_legal(Move(1, G4, F5)))

# ng4xe5 nf3xe5 nxe5 re1 f6 f4
# pos = Position.from_fen("r1bqk2r/ppp3pp/5p2/4n3/2Bp1P2/7P/PP1N2P1/R2QR1K1 b kq - 0 13")
# Evaluation(pos).init_attacks().evaluate(True)

# ng4xe5 nf3xe5 nxd5 re1 qe7
# pos = Position.from_fen("r1b1k2r/ppp1qppp/8/4n3/2Bp4/7P/PP1N1PP1/R2QR1K1 w kq - 2 13")
# Evaluation(pos).init_attacks().evaluate(True)

# import features
# pos = Position.from_fen("3rk2r/2P3p1/pn2p3/8/p4P2/2NqB2B/1P5p/R6K w k - 0 31")
# tf = features.ToFeature(pos).ann_features()
# print([len(a) for a in tf])
# print(tf)

import features
pos = Position.from_fen("1rq1kb1r/ppp2ppp/2np4/3N4/1PPpPn2/P4N1P/1B1Q1PP1/R4RK1 w k - 0 16")
import random
def goprofile():
    # now = time.time()
    # engine = Engine()
    # engine.debug = True
    # engine.infinite = False
    # engine.max_depth = 9
    # # pos = Position.from_fen("r3kbnr/ppp2ppp/2n5/8/2BP2b1/5N2/PP2KPPP/RNB4R w kq - 1 9")
    # pos = Position()
    # # print(pos)
    # # for i in range(100000):
    # #     sq = random.randint(0,63)
    # #     lowest_attacker(pos, 1<<sq)
    # # print(list(pos.generate_moves_all(legal=True)))
    # # print(engine.sort_moves(list(pos.generate_moves_all(legal=True)), pos, engine.si, 0, False))
    # engine.root_position = pos
    # engine.init_move_history()
    # engine.search_stats.time_start = time.time()
    # # print(engine.evaluate(pos))
    # # pos.toggle_side_to_move()
    # # print(engine.evaluate(pos))
    # pos.halfmove_clock = 50
    # engine.iterative_deepening()
    # # print(time.time() - now)
    for i in range(5000):
        tf = features.ToFeature(pos).ann_features()

# goprofile()

# pos = Position()
# pos.make_move(Move(Pt.N, G1, F3))
# print(pos.three_fold_hack)
# pos.make_move(Move(Pt.B_KNIGHT, G8, F6))
# print(pos.three_fold_hack)
# pos.make_move(Move(Pt.N, F3, G1))
# print(pos.three_fold_hack)
# pos.make_move(Move(Pt.B_KNIGHT, F6, G8))
# print(pos.three_fold_hack)

# pos.make_move(Move(Pt.N, B1, C3))

cProfile.run("goprofile()", filename="outprofile")
pstats.Stats("outprofile").strip_dirs().sort_stats("time").print_stats(15)
print()
pstats.Stats("outprofile").strip_dirs().sort_stats("cumulative").print_stats(15)

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
