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
    position.make_move(Move(PieceType.W_KNIGHT, B1, C3))
    position.make_move(Move(PieceType.B_KNIGHT, B8, C6))
    position.make_move(Move(PieceType.W_ROOK, A1, B1))
    assert(preserved_queenside_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(preserved_kingside_castle_rights(position.position_flags, Side.WHITE) == True)
    position.make_move(Move(PieceType.B_ROOK, A8, B8))
    assert(preserved_queenside_castle_rights(position.position_flags, Side.BLACK) == False)
    assert(preserved_kingside_castle_rights(position.position_flags, Side.BLACK) == True)
    position.make_move(Move(PieceType.W_KING, E1, G1))
    assert(preserved_queenside_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(preserved_kingside_castle_rights(position.position_flags, Side.WHITE) == False)
    position.make_move(Move(PieceType.B_KING, E8, G8))
    assert(preserved_queenside_castle_rights(position.position_flags, Side.BLACK) == False)
    assert(preserved_kingside_castle_rights(position.position_flags, Side.BLACK) == False)
    
def test_king_move():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN, E2, E4))
    position.make_move(Move(PieceType.B_PAWN, E7, E5))
    assert(preserved_kingside_castle_rights(position.position_flags, Side.WHITE) == True)
    assert(preserved_queenside_castle_rights(position.position_flags, Side.WHITE) == True)
    position.make_move(Move(PieceType.W_KING, E1, E2))
    assert(preserved_kingside_castle_rights(position.position_flags, Side.WHITE) == False)
    assert(preserved_queenside_castle_rights(position.position_flags, Side.WHITE) == False)
    
def test_position_1():
    position = Position()
    moves = []
    for move in list(position.generate_moves_all()):
        moves.append([move.piece_type, move.from_sq, move.to_sq])
    # print([Move(move[0], move[1], move[2]) for move in moves])
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
    
def test_zobrist2():
    pos = Position()
    pos.make_move(Move(Pt.P, E2, E4))
    zhash = pos.zobrist_hash
    assert(pos.en_pessant_sq == E3)
    pos2 = Position()
    pos2.make_move(Move(Pt.P, E2, E3))
    pos2.make_move(Move(Pt.BN, G8, F6))
    pos2.make_move(Move(Pt.B, F1, D3))
    pos2.make_move(Move(Pt.BN, F6, G8))
    pos2.make_move(Move(Pt.B, D3, E2))
    pos2.make_move(Move(Pt.BN, G8, F6))
    pos2.make_move(Move(Pt.B, E2, F1))
    pos2.make_move(Move(Pt.BN, F6, G8))
    pos2.make_move(Move(Pt.P, E3, E4))
    assert(pos2.en_pessant_sq is None)
    assert(pos2.zobrist_hash != zhash)
    pos3 = Position()
    pos3.make_move(Move(Pt.P, E2, E4))
    assert(pos3.zobrist_hash == zhash)
    
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
    rook_attacks = 0
    
    for r in iterate_pieces(position.pieces[PieceType.R]):
        att = rook_attack(r, position.occupied[0] | position.occupied[1])
        rook_attacks |= att
    assert(rook_attacks == 17011244761091413012)
    queen_attacks = 0
    for q in iterate_pieces(position.pieces[PieceType.Q]):
        queen_attacks |= queen_attack(q, position.occupied[0] | position.occupied[1])
    assert(queen_attacks == 1985618100175243261)
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
    q_att = 0
    for q in iterate_pieces(position.pieces[PieceType.B_QUEEN]):
        q_att |= queen_attack(q, position.occupied[0] | position.occupied[1])
    assert(q_att == 18232691494221090331)
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

