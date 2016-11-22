from search import *
from constants import *

# root_pos = Position()
# root = SearchPos(root_pos)
# val, final, child = negamax(root, -100000, 100000, 3, 1)
# val = -val

def play(position):
    root_pos = position or Position()
    root = SearchPos(root_pos)
    
    human_side = input("You play w or b? [w/b]")
    # while True:
    #     try:
    #         human_side = input("You play w or b? [w/b]")
    #         if human_side in ["w", "b"]:
    #             break
    #     except:
    #         pass
    
    if (human_side == "w" and root_pos.side_to_move() == Side.BLACK.value) or \
       (human_side == "b" and root_pos.side_to_move() == Side.WHITE.value):
        val, chosen = iterative_deepening(4, root)
        print("==== Computer move: ==========")
        print(chosen)
        print("==============================")
        root.position.make_move(chosen)
        
    while not root.position.is_mate():
        user_move = get_user_move(root.position.side_to_move())
        # while True:
        #     try:
        #         user_move = get_user_move()
        #     except:
        #         pass
        print("==== Human move: ==============")
        print(user_move)
        print("===============================")
        root.position.make_move(user_move)
        
        val, chosen = iterative_deepening(4, root)
        print("==== Computer move: ==========")
        print(chosen)
        print("==============================")
        root.position.make_move(chosen)

    print("Mate?:", child.position.is_mate())
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

pos = Position()

# alina's game with bxg4 wins piece
pos = Position.from_fen("r2qk2r/1pp2p2/p1n1b2p/2b3p1/3pP1n1/N2P4/PPPNBPPP/R1BQ1RK1 w kq - 0 0")
# pos = Position.from_fen("r2qkb1r/1pp2p2/p1n1b2p/6p1/3pP1n1/3P4/PPPNBPPP/R1BQ1RK1 b kq - 0 1")
# pos = Position.from_fen("2r1k1nr/2p2p2/2n1bq1p/ppb1p3/2NpP3/2PP4/PP1BBPPN/R2Q1RK1 w k - 0 17")
# pos = Position.from_fen("2r1k1nr/2p2p2/4b2p/npq1p3/3PP3/3P4/P2BBPPN/R2Q1RK1 b k - 0 20")
# pos.make_move(Move(PieceType.B_QUEEN.value, C5, A3))
# pos.make_move(Move(PieceType.W_QUEEN.value, D1, B1))
# pos = Position.from_fen("r2qkbnr/pp2pppp/n1p5/3pP3/3P2b1/N1PB1N2/PP3PPP/R1BQ1RK1 b kq - 0 0")
# pos = Position.from_fen("r1b1kb1r/1pp1pppp/p1n2n2/8/3P2q1/2NB1N2/PPP2PPP/R1BQR1K1 b kq - 1 8")

# play(pos)
# pos.make_move(Move(PieceType.W_BISHOP.value, E2, G4))
# pos.make_move(Move(PieceType.B_BISHOP.value, E6, G4))
# p = SearchPos(pos)
# print_moves(p.children([SearchInfo()], 0))

import time
now = time.time()
val = iterative_deepening(.95, SearchPos(pos))
print(val / 200)
then = time.time()
print(then-now, 's')

# def do_test():
#     pos = Position.from_fen("r1b1kb1r/1pp1pppp/p1n2n2/8/3P2q1/2NB1N2/PPP2PPP/R1BQR1K1 b kq - 1 8")
#     pos.make_move(Move(PieceType.B_KNIGHT.value, C6, D4))
#     return pos

# import timeit
# print("evaluate(pos)")
# timeit.timeit("evaluate(pos)", setup="from search import evaluate; from __main__ import do_test; pos=do_test()", number=1000)
# print("eval_see(pos, Move(2,F3,D4))")
# timeit.timeit("eval_see(pos, Move(2,F3,D4))", setup="from evals import eval_see; from move import Move; from bb import F3, D4; from __main__ import do_test; pos=do_test()", number=1000)
