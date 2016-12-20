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

# pos = Position()

pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")
pos = Position.from_fen("r1bqk2r/ppp3pp/5p2/4n3/2Bp1P2/7P/PP1N2P1/R2QR1K1 b kq - 0 13")
evaluate(pos, True)

# import time 
# now = time.time()
# val, si = iterative_deepening(4.5, pos)
# # val = search(SearchPos(pos), [None] * 64, 0, -10000000, 10000000, 1, .001, True)
# # print("node count", node_count)
# print(val / 200)
# then = time.time()
# print(then-now, 's')

# pos = Position.from_fen("rnb1kbnr/pp1ppppp/2p5/q7/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3")
# pos.make_move(Move(PieceType.K,E1,E2))
# evaluate(pos, True)

# pos = Position.from_fen("rnbqkb1r/ppp1pp1p/5np1/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 4")
# evaluate(pos, True)
# pos.make_move(Move(PieceType.R,H1,G1))
# print()
# evaluate(pos, True)

# pos = Position.from_fen("r1bqk2r/ppp2ppp/8/4n3/2Bp4/7P/PP1N1PP1/R2QR1K1 b kq - 1 12")
# evaluate(pos, True)
# print()
# pos.make_move(Move(PieceType.Q + 6, D8, E7))
# evaluate(pos, True)
# print()
# pos = Position.from_fen("r1bqk2r/ppp2ppp/8/4n3/2Bp4/7P/PP1N1PP1/R2QR1K1 b kq - 1 12")
# pos.make_move(Move(PieceType.Q + 6, D8, F6))
# evaluate(pos, True)
# pos.make_move(Move(PieceType.N, D2, F3))
# evaluate(pos, True)

# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")
# pos = Position.from_fen("r1b1k2r/ppp2ppp/5q2/4n3/2Bp4/5N1P/PP3PP1/R2QR1K1 b kq - 3 13")
# pos = Position.from_fen("r1bq1k1r/ppp3pp/5p2/4n3/2BN4/7P/PP3PP1/R2QR1K1 b - - 0 14")
# pos = Position.from_fen("r1bqk2r/1pp2ppp/2n5/4P3/p1BN4/7P/PP1Q1PP1/3R1RK1 b kq - 0 15")

# pos = Position.from_fen("r1bqk2r/1pp2ppp/8/4n3/p1BN4/7P/PP1Q1PP1/3RR1K1 b kq - 1 16")
# evaluate(pos, True)

#  after g5
# pos = Position.from_fen("r1bqk2r/ppp2p1p/2n4n/4P1p1/2Bp4/5N1P/PP1N1PP1/R2QR1K1 w kq - 0 12")
# evaluate(pos, True)
# print()
# # after 00
# pos = Position.from_fen("r1bq1rk1/ppp2ppp/2n4n/4P3/2Bp4/5N1P/PP1N1PP1/R2QR1K1 w - - 4 12")
# evaluate(pos, True)

# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n4n/4P3/2Bp4/5N1P/PP1N1PP1/R2QR1K1 b kq - 3 11")
# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")

# f6 f4
# pos = Position.from_fen("r1bqk2r/ppp3pp/5p2/4n3/2Bp1P2/7P/PP1N2P1/R2QR1K1 b kq - 0 13")
# evaluate(pos, True)

# # f6 f4 f5 
# pos = Position.from_fen("r1bqk2r/ppp3pp/8/4np2/2Bp1P2/7P/PP1N2P1/R2QR1K1 w kq - 0 14")
# evaluate(pos, True)

# # f6 f4 c6
# pos = Position.from_fen("r1bqk2r/pp4pp/2p2p2/4n3/2Bp1P2/7P/PP1N2P1/R2QR1K1 w kq - 0 14")
# evaluate(pos, True)

# play(pos)

#Ng4-e5 Re1
# pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4n3/2Bp4/5N1P/PP1N1PP1/R2QR1K1 b kq - 1 11")
# evaluate(pos, True)

#Ng4-e5 Ne5 Ne5 Re1
# pos = Position.from_fen("r1bqk2r/ppp2ppp/8/4n3/2Bp4/7P/PP1N1PP1/R2QR1K1 b kq - 1 12")
# evaluate(pos, True)

# import time 
# now = time.time()
# val, si = iterative_deepening(4.5, pos)
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
