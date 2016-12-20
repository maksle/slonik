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

import time 
now = time.time()
val, si = iterative_deepening(4.5, pos)
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
