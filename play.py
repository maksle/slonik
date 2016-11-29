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
    
    if (human_side == "w" and root.side_to_move() == Side.BLACK) or \
       (human_side == "b" and root.side_to_move() == Side.WHITE):
        val, si = iterative_deepening(5, root, si)
        # print("==== Computer move: ==========")
        # print(chosen)
        # print("==============================")
        root.make_move(si[0].pv[0])
        
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
        
        val, si = iterative_deepening(5, root, si)
        # print("==== Computer move: ==========")
        # print("==============================")
        root.make_move(si[0].pv[0])

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

pos = Position()

# alina's game with bxg4 wins piece
# pos = Position.from_fen("r2qk2r/1pp2p2/p1n1b2p/2b3p1/3pP1n1/N2P4/PPPNBPPP/R1BQ1RK1 w kq - 0 0")
# pos = Position.from_fen("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1")
# pos = Position.from_fen("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 1 1")

# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 1")
# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4n3/2Bp4/5N1P/PP1N1PP1/R2Q1RK1 w kq - 0 2")
# pos = Position.from_fen("r1bqk2r/ppp2ppp/8/4n3/2Bp4/7P/PP1N1PP1/R2Q1RK1 w kq - 0 3")

# pos = Position.from_fen("r1bqk2r/ppp2ppp/8/4n3/2Bp4/7P/PP1N1PP1/R2Q1RK1 w kq - 1 1")
# pos = Position.from_fen("1Qb1kb1r/3q1pp1/p1Rp2n1/4p2p/2PNPP2/N7/6PP/1B4K1 b k - 1 31")

play(pos)

# pos.make_move(Move(1,E2,E4))
# pos.make_move(Move(8,B8,C6))
# pos.make_move(Move(1,D2,D3))
# evaluate(pos, True)
# print()

# pos = Position()
# pos.make_move(Move(1,E2,E4))
# pos.make_move(Move(8,B8,C6))
# pos.make_move(Move(1,D2,D4))
# evaluate(pos, True)

# pos = Position.from_fen("rnbqkbnr/ppp1pppp/3p4/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
# print(list(pos.generate_moves()))

# pos = Position.from_fen("rnbqkbnr/1p1p1pp1/7p/p1pP4/4P3/2N2N2/PP2BPPP/R1BQK2R b KQkq - 0 8")
# print("eval", evaluate(pos, True))
# pos.make_move(Move(PieceType.B_KING, E8, E7))
# print()
# print("eval", evaluate(pos, True))

# pos = Position.from_fen("r1bqk2r/pp3ppp/8/2p1n3/2Bp4/7P/PP1N1PP1/1R1Q1RK1 w kq - 0 2")
# print_bb(all_pawn_attack_potentials(pos, 0))
# return count_bits(potential ^ FULL_BOARD) * 20
# print()
# print_bb(all_pawn_attack_potentials(pos, 1))

# import time 
# now = time.time()
# val, si = iterative_deepening(5, pos)
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

# def do_test():
#     pos = Position.from_fen("r1b1kb1r/1pp1pppp/p1n2n2/8/3P2q1/2NB1N2/PPP2PPP/R1BQR1K1 b kq - 1 8")
#     pos.make_move(Move(PieceType.B_KNIGHT, C6, D4))
#     return pos

# import timeit
# print("evaluate(pos)")
# timeit.timeit("evaluate(pos)", setup="from search import evaluate; from __main__ import do_test; pos=do_test()", number=1000)
# print("eval_see(pos, Move(2,F3,D4))")
# timeit.timeit("eval_see(pos, Move(2,F3,D4))", setup="from evals import eval_see; from move import Move; from bb import F3, D4; from __main__ import do_test; pos=do_test()", number=1000)

# 1262 nodes, .05, 3.6s avg 3.3, max 9
# 7987 nodes, .01, 21.8s avg 4.9, max 12
# 20723 nodes, .005, 56.8s, avg 5.8, max 19
# 94124 nodes, .001, 257.6s, avg 7.8, max 21
