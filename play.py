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
        val, chosen = iterative_deepening(3, root)
        print("Computer move:")
        print(chosen)
        root.position.make_move(chosen)
        
    while not root.position.is_mate():
        user_move = get_user_move(root.position.side_to_move())
        # while True:
        #     try:
        #         user_move = get_user_move()
        #     except:
        #         pass
        print("Human move:")
        print(user_move)
        print()
        root.position.make_move(user_move)
        
        val, chosen = iterative_deepening(3, root)
        print("Computer move:")
        print(chosen)
        print()
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

# pos.make_move(Move(PieceType.W_PAWN.value, E2, E4))
# pos.make_move(Move(PieceType.B_PAWN.value, D7, D5))
# pos.make_move(Move(PieceType.W_KING.value, E1, E2))
# pos.make_move(Move(PieceType.W_PAWN.value, D2, D3))
# pos.make_move(Move(PieceType.B_KNIGHT.value, G8, H6))

# evaluate(pos)
# print(evaluate_moves(pos))

# pos.make_move(Move(PieceType.B_PAWN.value, C7, C5))
# pos.make_move(Move(PieceType.W_KNIGHT.value, B1, C3))
# print_moves(list(pos.generate_moves()))

# print_moves(list(pos.generate_moves()))
# print_moves(pos.moves)

# alina's game with bxg4 wins piece
# pos = Position.from_fen("r2qk2r/1pp2p2/p1n1b2p/2b3p1/3pP1n1/N2P4/PPPNBPPP/R1BQ1RK1 w kq - 0 0")
# pos = Position.from_fen("r2qkb1r/1pp2p2/p1n1b2p/6p1/3pP1n1/3P4/PPPNBPPP/R1BQ1RK1 b kq - 0 1")
# pos = Position.from_fen("2r1k1nr/2p2p2/2n1bq1p/ppb1p3/2NpP3/2PP4/PP1BBPPN/R2Q1RK1 w k - 0 17")
pos = Position.from_fen("2r1k1nr/2p2p2/4b2p/npq1p3/3PP3/3P4/P2BBPPN/R2Q1RK1 b k - 0 20")
pos.make_move(Move(PieceType.B_QUEEN.value, C5, A3))
pos.make_move(Move(PieceType.W_QUEEN.value, D1, B1))

# play(pos)

import time
now = time.time()
val,c = iterative_deepening(5, SearchPos(pos))
# val = -val
print(val / 200, c)
then = time.time()
print(then-now, 's')
