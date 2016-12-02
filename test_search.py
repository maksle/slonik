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
    
test_lowest_attacker()
    
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

