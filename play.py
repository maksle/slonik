from search import *
from constants import *

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

pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")

# ng4xe5 nf3xe5 nxe5 re1 f6 f4
# pos = Position.from_fen("r1bqk2r/ppp3pp/5p2/4n3/2Bp1P2/7P/PP1N2P1/R2QR1K1 b kq - 0 13")
# Evaluation(pos).init_attacks().evaluate(True)

# ng4xe5 nf3xe5 nxd5 re1 qe7
# pos = Position.from_fen("r1b1k2r/ppp1qppp/8/4n3/2Bp4/7P/PP1N1PP1/R2QR1K1 w kq - 2 13")
# Evaluation(pos).init_attacks().evaluate(True)

import time 
now = time.time()
val, si = iterative_deepening(3, pos)
# val = search(SearchPos(pos), [None] * 64, 0, -10000000, 10000000, 1, .001, True)
# print("node count", node_count)
print(val / 200)
then = time.time()
print(then-now, 's')

# import time
# now = time.time()
# perft(Position(Position.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0")), 3, True)
# then = time.time()
# print(then-now, end='s')

# python3 -m cProfile -o profile5 play.py
# pstats.Stats('profile5').strip_dirs().sort_stats('cumulative').print_stats(15)
