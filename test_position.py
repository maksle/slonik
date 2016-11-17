from position import *
from evals import *

def test_mobility():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN.value, E2, E4))
    assert(mobility(position, Side.WHITE.value) == 23)
    position.make_move(Move(PieceType.B_KNIGHT.value, G8, F6))
    position.make_move(Move(PieceType.W_PAWN.value, C2, C3))
    assert(mobility(position, Side.WHITE.value) == 23)
    assert(mobility(position, Side.BLACK.value) == 14)
    # print(mobility(position, Side.BLACK.value))
    
def test_castle():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN.value, E2, E4))
    position.make_move(Move(PieceType.B_PAWN.value, E7, E5))
    position.make_move(Move(PieceType.W_KNIGHT.value, G1, F3))
    position.make_move(Move(PieceType.B_KNIGHT.value, G8, F6))
    position.make_move(Move(PieceType.W_BISHOP.value, F1, C4))
    position.make_move(Move(PieceType.B_BISHOP.value, F8, C5))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE.value) == True)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK.value) == True)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.W_KING.value, E1, G1))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE.value) == False)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK.value) == True)
    assert(position.w_king_castle_ply == 7)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.B_KING.value, E8, G8))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE.value) == False)
    assert(preserved_castle_rights(position.position_flags, Side.BLACK.value) == False)
    assert(position.w_king_castle_ply == 7)
    assert(position.b_king_castle_ply == 8)
    
def test_king_move():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN.value, E2, E4))
    position.make_move(Move(PieceType.B_PAWN.value, E7, E5))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE.value) == True)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)
    position.make_move(Move(PieceType.W_KING.value, E1, E2))
    assert(preserved_castle_rights(position.position_flags, Side.WHITE.value) == False)
    assert(position.w_king_castle_ply == -1)
    assert(position.b_king_castle_ply == -1)

def test_move_piece_attacks():
    """Tests that piece attacks are correctly updated after piece move"""
    position = Position()
    position.make_move(Move(PieceType.W_PAWN.value, E2, E4))
    assert(position.piece_attacks[PieceType.W_BISHOP.value] == 0x804020105a00)
    
    # position.make_move(Move(PieceType.B_KNIGHT.value, G8, F6))
    # position.make_move(Move(PieceType.W_QUEEN.value, D1, E2))
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
    position.make_move(Move(PieceType.W_KNIGHT.value, G1, F3))
    assert(zhash != position.zobrist_hash)
    position.make_move(Move(PieceType.B_KNIGHT.value, G8, F6))
    position.make_move(Move(PieceType.W_KNIGHT.value, F3, G1))
    assert(zhash != position.zobrist_hash)
    position.make_move(Move(PieceType.B_KNIGHT.value, F6, G8))
    assert(zhash == position.zobrist_hash)
    
def test_move():
    move = Move(PieceType.W_KNIGHT.value, G1, F3)
    move2 = Move.move_uncompacted(move.compact())
    assert(move.from_sq == move2.from_sq)
    assert(move.to_sq == move2.to_sq)
    assert(move.piece_type == move2.piece_type)
    assert(move.promo_piece == move2.promo_piece)
    # print(move, move2)

    
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

