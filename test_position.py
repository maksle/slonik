from position import *
from evals import *

def test_mobility():
    # position = Position()
    # position.make_move(Move(PieceType.W_PAWN, E2, E4))
    # assert(mobility(position, Side.WHITE, []) == 23)
    # position.make_move(Move(PieceType.B_KNIGHT, G8, F6))
    # position.make_move(Move(PieceType.W_PAWN, C2, C3))
    # assert(mobility(position, Side.WHITE) == 23)
    # assert(mobility(position, Side.BLACK) == 14)
    # print(mobility(position, Side.BLACK))
    pass
    
def test_castle():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN, E2, E4))
    position.make_move(Move(PieceType.B_PAWN, E7, E5))
    position.make_move(Move(PieceType.W_KNIGHT, G1, F3))
    position.make_move(Move(PieceType.B_KNIGHT, G8, F6))
    position.make_move(Move(PieceType.W_BISHOP, F1, C4))
    position.make_move(Move(PieceType.B_BISHOP, F8, C5))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE) == True)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK) == True)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.W_KING, E1, G1))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK) == True)
    assert(position.w_king_castle_ply == 6)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.B_KING, E8, G8))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK) == False)
    assert(position.w_king_castle_ply == 6)
    assert(position.b_king_castle_ply == 7)
    
def test_king_move():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN, E2, E4))
    position.make_move(Move(PieceType.B_PAWN, E7, E5))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE) == True)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.W_KING, E1, E2))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)

def test_move_piece_attacks():
    """Tests that piece attacks are correctly updated after piece move"""
    position = Position()
    position.make_move(Move(PieceType.W_PAWN, E2, E4))
    assert(position.piece_attacks[PieceType.W_BISHOP] == 0x804020105a00)
    
    # position.make_move(Move(PieceType.B_KNIGHT, G8, F6))
    # position.make_move(Move(PieceType.W_QUEEN, D1, E2))
    # for pt in PieceType.piece_types(base_only=False):
    #     print("pt is ", pt, HUMAN_PIECE[pt])
    #     attacks = position.piece_attacks[pt]
    #     print_bb(attacks)
    #     print()
    
def test_position_1():
    position = Position()
    moves = []
    for move in list(position.generate_moves()):
        moves.append([move.piece_type, move.from_sq, move.to_sq])
    assert len(moves) == 20
    
def test_zobrist():
    position = Position()
    zhash = position.zobrist_hash
    position.make_move(Move(PieceType.W_KNIGHT, G1, F3))
    assert(zhash != position.zobrist_hash)
    position.make_move(Move(PieceType.B_KNIGHT, G8, F6))
    position.make_move(Move(PieceType.W_KNIGHT, F3, G1))
    assert(zhash != position.zobrist_hash)
    position.make_move(Move(PieceType.B_KNIGHT, F6, G8))
    assert(zhash == position.zobrist_hash)
    
def test_move():
    move = Move(PieceType.W_KNIGHT, G1, F3)
    move2 = Move.move_uncompacted(move.compact())
    assert(move.from_sq == move2.from_sq)
    assert(move.to_sq == move2.to_sq)
    assert(move.piece_type == move2.piece_type)
    assert(move.promo_piece == move2.promo_piece)
    # print(move, move2)

def test_fen_zobrist():
    position = Position()
    position2 = Position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0")
    assert(position.zobrist_hash == position2.zobrist_hash)

def test_promotion():
    position = Position.from_fen("3R4/1k3P2/8/5R2/8/8/6K1/6Q1 w - - 1 3")
    move = Move(PieceType.P, F7, F8, MoveType.promo, PieceType.Q)
    position.make_move(move)
    assert(position.occupied[0] == 1441151897938428418)
    assert(position.squares[bit_position(F8)] == PieceType.Q)
    assert(position.piece_attacks[PieceType.R] == 17011244761091413012)
    assert(position.piece_attacks[PieceType.Q] == 1985618100175243261)
    assert(position.pieces[PieceType.Q] == (F8 | G1))
    assert(position.pieces[PieceType.P] == 0)
    position2 = Position.from_fen("3R1Q2/1k6/8/5R2/8/8/6K1/6Q1 b - - 1 3")
    assert(position.zobrist_hash == position2.zobrist_hash)
    
def test_promotion2():
    position = Position.from_fen("6q1/6k1/8/8/5r2/8/1K3p2/3r4 b - - 1 3")
    move = Move(PieceType.B_PAWN, F2, F1, MoveType.promo, PieceType.B_QUEEN)
    position.make_move(move)
    assert(position.occupied[1] == 144678138096386068)
    assert(position.squares[bit_position(F1)] == PieceType.B_QUEEN)
    assert(position.piece_attacks[PieceType.B_QUEEN] == 18232691494221090331)
    assert(position.pieces[PieceType.B_QUEEN] == (F1 | G8))
    
if __name__ == "__main__":
    import sys
    from inspect import getmembers, isfunction

    THIS_MODULE = sys.modules[__name__]
    TEST_FUNCS = [o[1] for o in getmembers(THIS_MODULE)
                  if isfunction(o[1])
                  and o[0].startswith('test_')]

    for test_func in TEST_FUNCS:
        test_func()

    print('{0} tests passed'.format(len(TEST_FUNCS)))

