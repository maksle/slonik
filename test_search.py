from search import *

def test_lowest_attacker():
    position = Position()
    position.make_move(Move(PieceType.W_PAWN, E2, E4))
    position.make_move(Move(PieceType.B_PAWN, D7, D5))
    position.make_move(Move(PieceType.W_KNIGHT, B1, C3))
    position.make_move(Move(PieceType.B_KNIGHT, G8, F6))
    position.make_move(Move(PieceType.W_PAWN, F2, F3))
    
    # print(eval_see(position, E4))
    # print(lowest_attacker(position,E4))
    assert(lowest_attacker(position, E4)[1] == 68719476736)
    
    move = Move(PieceType.B_PAWN, D5, E4)
    assert(eval_see(position, move) == 0)
    position.make_move(move)
    assert(lowest_attacker(position, E4)[1]) == 262144

    move = Move(PieceType.W_PAWN, F3, E4)
    assert(eval_see(position, move) > 0)
    position.make_move(move)

def test_generate_moves_all():
    pos = Position()
    moves = pos.generate_moves_all()
    assert len(list(moves)) == 20

def test_generate_moves_all_2():
    pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")
    moves = pos.generate_moves_all()
    assert len(list(moves)) == 39

def test_generate_moves_violent():
    pos = Position.from_fen("r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10")
    moves = pos.generate_moves_violent()
    assert len(list(moves)) == 3

def test_generate_moves_violent_2():
    pos = Position.from_fen("3b4/1k2P3/8/8/8/5N2/6K1/8 w - - 1 10")
    moves = pos.generate_moves_violent()
    # print_moves(moves)
    assert len(list(moves)) == 8

def test_genrate_moves_violent_3():
    pos = Position.from_fen("8/1k6/8/5Pp1/8/8/6K1/8 w - g6 1 10")
    moves = pos.generate_moves_violent()
    # print_moves(moves)
    assert len(list(moves)) == 1

def test_generate_moves_in_check():
    pos = Position.from_fen("8/1k6/8/3r4/8/NR2b3/6K1/8 b - - 1 10")
    moves = pos.generate_moves_in_check()
    assert len(list(moves)) == 10

def test_generate_moves_in_check_2():
    # double check
    pos = Position.from_fen("8/1k6/3N4/3r4/8/1R2b3/6K1/8 b - - 1 10")
    moves = pos.generate_moves_in_check()
    # print_moves(moves)
    assert len(list(moves)) == 8

def test_generate_moves_in_check_2():
    # en-pessant to stop check
    pos = Position.from_fen("8/1k6/8/5Pp1/5K2/8/8/8 w - g6 1 10")
    moves = pos.generate_moves_in_check()
    # print_moves(moves)
    assert len(list(moves)) == 8

def test_generate_moves_in_check_2():
    pos = Position.from_fen("8/1k6/5K1r/5Pp1/8/8/8/8 w - g6 1 10")
    moves = pos.generate_moves_in_check()
    # print_moves(moves)
    assert len(list(moves)) == 8

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

