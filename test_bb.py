from bb import *
from print_bb import *
from move_gen import *
from piece_type import *

def test_north_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11110111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = north_attack(attacker, free)
    # print_bb(res,)
    assert res == 2260630266445824

def test_nw_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '10111111',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = nw_attack(attacker, free)
    # print_bb(res)
    assert res == 18049651601047552

def test_nw_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = nw_attack(attacker, free)
    # print_bb(res)
    assert res == 9241421688455823360


def test_ne_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111110',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = ne_attack(attacker, free)
    # print_bb(res)
    assert res == 283691179835392

def test_ne_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
        '11111111',
        '10111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = ne_attack(attacker, free)
    # print_bb(res)
    assert res == 283691179835392

def test_sw_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11110111',
        '11111111',
        '11011111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = sw_attack(attacker, free)
    # print_bb(res)
    assert res == 1056768

def test_east_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11011111',
        '11111111',
        '10110110',
        '11111111',
        '11011111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = east_attack(attacker, free)
    # print_bb(res)
    assert res == 117440512

def test_queen_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = queen_attack(attacker, free)
    assert res == 9820426766351346249

def test_queen_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000010',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = queen_attack(attacker, free)
    # print_bb(res)
    # print(res)
    assert res == 11118032872913386091

def test_rook_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = rook_attack(attacker, free)
    # print_bb(res)
    # print(res)
    assert res == 578721386714368008

def test_bishop_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = bishop_attack(attacker, free)
    # print_bb(res)
    assert res == 9241705379636978241


def test_bishop_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000010',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
        '11111111',
    ]
    attacker = int(''.join(g_arr), 2)
    free = int(''.join(p_arr), 2)

    res = bishop_attack(attacker, free)
    # print_bb(res)
    # print()
    assert res == 0x2010080500050810
    
def test_knight_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 22136263676928

def test_knight_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000010',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 5531918402816

def test_knight_attack_3():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '01000000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 175990581010432

def test_knight_attack_4():
    g_arr = [
        '00000000',
        '01000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 1152939783987658752

def test_knight_attack_5():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000100',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 168886289

def test_knight_attack_6():
    g_arr = [
        '00000000',
        '01000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000100',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = knight_attack(attacker)
    # print_bb(res)
    assert res == 0x101432a02a150011
    
def test_king_attack():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = king_attack(attacker)
    # print_bb(res)
    assert res == 120596463616


def test_king_attack_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = king_attack(attacker)
    # print_bb(res)
    assert res == 7188

def test_king_attack_3():
    g_arr = [
        '10000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = king_attack(attacker)
    # print_bb(res)
    assert res == 2365848970648778440704

def test_king_attack_4():
    g_arr = [
        '00000001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = king_attack(attacker)
    # print_bb(res)
    assert res == 18591703686715539456

def test_pawn_attack():
    g_arr = [
        '00000000',
        '00000100',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00100000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = pawn_attack(attacker, Side.WHITE.value)
    # print_bb(res)
    assert res == 720576026283868160

def test_pawn_attack_2():
    g_arr = [
        '00000000',
        '00000010',
        '00000000',
        '00000001',
        '10001000',
        '00000000',
        '00100000',
        '00000000',
    ]
    attacker = int(''.join(g_arr), 2)
    res = pawn_attack(attacker, Side.WHITE.value)
    # print_bb(res)
    assert res == 0x500025400500000
    
def test_iterate_pieces():
    g_arr = [
        '00000000',
        '00000100',
        '00000000',
        '00000000',
        '00001000',
        '00000000',
        '00100000',
        '00000000',
    ]
    pieces = int(''.join(g_arr), 2)
    # for piece in next_piece(pieces):
    #     print_bb(piece)
    #     print()
    # print(list(map(hex, next_piece(pieces))))
    assert list(iterate_pieces(pieces)) == [0x2000, 0x8000000, 0x4000000000000]

def test_iterate_pieces_2():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
    ]
    pieces = int(''.join(g_arr), 2)
    assert list(iterate_pieces(pieces)) == []

def test_knight_moves():
    g_arr = [
        '00000000',
        '00000100',
        '00000000',
        '00000000',
        '00010000',
        '00000000',
        '00000000',
        '00000000',
    ]
    p_arr = [
        '00000001',
        '00000000',
        '00100000',
        '00000000',
        '00000000',
        '00000100',
        '00000000',
        '00000000',
    ]
    knights = int(''.join(g_arr), 2)
    own = int(''.join(p_arr), 2)
    # print(list(move for move in knight_moves(knights, own)))
    assert list(move for move in knight_moves(knights, own)) == [(268435456, 2048), (268435456, 8192), (268435456, 4194304), (268435456, 17179869184), (268435456, 274877906944), (268435456, 8796093022208), (1125899906842624, 8589934592), (1125899906842624, 34359738368), (1125899906842624, 1099511627776), (1125899906842624, 17592186044416), (1125899906842624, 1152921504606846976)]
    # for move in (move for move in knight_moves(knights, own)):
        # print('From:')
        # print_bb(move[0])
        # print('To:')
        # print_bb(move[1])
        # print()

def test_rook_move():
    g_arr = [
        '00000000',
        '00000100',
        '00000000',
        '00000000',
        '00010100',
        '00000000',
        '00000000',
        '00000000',
    ]
    own_arr = [
        '10000000',
        '00000100',
        '00000100',
        '00000000',
        '00010100',
        '00000000',
        '00000000',
        '00010000',
    ]
    other_arr = [
        '00000000',
        '00100000',
        '00000001',
        '00000000',
        '00000001',
        '00000000',
        '00000000',
        '00000000',
    ]
    rooks = int(''.join(g_arr), 2)
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    assert list(move for move in rook_moves(rooks, own, other)) == [(67108864, 4), (67108864, 1024), (67108864, 262144), (67108864, 16777216), (67108864, 33554432), (67108864, 134217728), (67108864, 17179869184), (268435456, 4096), (268435456, 1048576), (268435456, 134217728), (268435456, 536870912), (268435456, 1073741824), (268435456, 2147483648), (268435456, 68719476736), (268435456, 17592186044416), (268435456, 4503599627370496), (268435456, 1152921504606846976), (1125899906842624, 281474976710656), (1125899906842624, 562949953421312), (1125899906842624, 2251799813685248), (1125899906842624, 4503599627370496), (1125899906842624, 9007199254740992), (1125899906842624, 288230376151711744)]
    # print(list(move for move in rook_moves(rooks, own, other)))
    # for move in (move for move in rook_moves(rooks, own, other)):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()   

def test_bishop_move():
    g_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00010010',
        '00000000',
        '00000000',
        '00000000',
    ]
    own_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00010010',
        '00000000',
        '00000000',
        '00000000',
    ]
    other_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00100000',
        '00000000',
        '00000000',
    ]
    rooks = int(''.join(g_arr), 2)
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    # print(list(move for move in rook_moves(rooks, own, other)))
    # for move in (move for move in bishop_moves(rooks, own, other)):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()   


def test_pawn_moves():
    pawns_arr = [
        '00000000',
        '10000001',
        '00000000',
        '00110000',
        '00000000',
        '01000100',
        '01100011',
        '00000000',
    ]
    own_arr = [
        '00000000',
        '10000001',
        '00000000',
        '00110000',
        '00000000',
        '01000100',
        '01100011',
        '01000000',
    ]
    other_arr = [
        '00000001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00100001',
        '00000000',
        '00000000',
    ]
    pawns = int(''.join(pawns_arr), 2)
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    e7 = 0x8000000000000
    e5 = 0x800000000
    assert list(pawn_moves(pawns, own, other,
               Side.WHITE.value, PieceType.B_PAWN.value, e7, e5)) == [(512, 65536), (512, 131072), (512, 33554432), (16384, 2097152), (262144, 67108864), (4194304, 1073741824), (68719476736, 8796093022208), (68719476736, 17592186044416), (137438953472, 35184372088832), (36028797018963968, 9223372036854775808)]

    # pawn_moves(pawns, own, other,
    #            Side.WHITE.value, PieceType.B_PAWN.value, e7, e5)
    # print(list(move for move in pawn_moves(pawns, own, other,
    #            Side.WHITE.value, PieceType.B_PAWN.value, e7, e5)))
    # for move in (move for move in pawn_moves(pawns, own, other,
    #            Side.WHITE.value, PieceType.B_PAWN.value, e7, e5)):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_moves():
    king_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001000',
    ]
    own_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00001100',
        '00011000',
    ]
    attacked_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000100',
        '00000100',
    ]
    king = int(''.join(king_arr), 2)
    own = int(''.join(own_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
   
    # print(list(king_moves(king, own, attacked)))
    assert list(king_moves(king, own, attacked)) == [(8, 4096)]
    # for move in king_moves(king, own, attacked):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_castle_moves():
    own_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000111',
        '10011001',
    ]
    other_arr = [
        '10001001',
        '11100011',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacked_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
    ]
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
    position_flags = Side.WHITE.value << 6

    # king_castle_moves(own, other, attacked, position_flags)
    # print("Castles:")
    # print(list(king_castle_moves(own, other, attacked, position_flags)))

    assert list(king_castle_moves(own, other, attacked, position_flags)) == [(8, 2)]

    # for move in king_castle_moves(own, other, attacked, position_flags):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_castle_moves_2():
    own_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000111',
        '10001001',
    ]
    other_arr = [
        '10001001',
        '11100011',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
        '00000000',
        '00000000',
    ]
    attacked_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
    ]
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
    position_flags = Side.WHITE.value << 6

    # king_castle_moves(own, other, attacked, position_flags)
    # print(list(king_castle_moves(own, other, attacked, position_flags)))
    assert list(king_castle_moves(own, other, attacked, position_flags)) == [(8, 2), (8, 32)]

    # for move in king_castle_moves(own, other, attacked, position_flags):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_castle_moves_3():
    own_arr = [
        '10001001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000111',
        '00000000',
    ]
    other_arr = [
        '00000000',
        '11100011',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
        '00000000',
        '10001001',
    ]
    attacked_arr = [
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
    ]
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
    position_flags = Side.BLACK.value << 6

    # king_castle_moves(own, other, attacked, position_flags)
    # print(list(king_castle_moves(own, other, attacked, position_flags)))
    assert list(king_castle_moves(own, other, attacked, position_flags)) == [(8<<56, 2<<56), (8<<56, 32<<56)]

    # for move in king_castle_moves(own, other, attacked, position_flags):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_castle_moves_4():
    own_arr = [
        '10001001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000111',
        '00000000',
    ]
    other_arr = [
        '00000000',
        '11100011',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
        '00000000',
        '10001001',
    ]
    attacked_arr = [
        '00100001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
    ]
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
    position_flags = Side.BLACK.value << 6

    # king_castle_moves(own, other, attacked, position_flags)
    # print(list(king_castle_moves(own, other, attacked, position_flags)))
    assert list(king_castle_moves(own, other, attacked, position_flags)) == [(8<<56, 2<<56)]

    position_flags = Side.BLACK.value | 2
    # print('here', (own, other, attacked, position_flags))
    # print(list(king_castle_moves(own, other, attacked, position_flags)))
    assert list(king_castle_moves(own, other, attacked, position_flags)) == []
    
    # for move in king_castle_moves(own, other, attacked, position_flags):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

def test_king_castle_moves_5():
    own_arr = [
        '10001001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000111',
        '00000000',
    ]
    other_arr = [
        '00000000',
        '11100011',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
        '00000000',
        '10001001',
    ]
    attacked_arr = [
        '00100001',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00000000',
        '00011111',
        '00000000',
    ]
    own = int(''.join(own_arr), 2)
    other = int(''.join(other_arr), 2)
    attacked = int(''.join(attacked_arr), 2)
    position_flags = Side.BLACK.value << 6

    # king_castle_moves(own, other, attacked, position_flags)
    # print(list(king_castle_moves(own, other, attacked, position_flags)))
    assert list(king_castle_moves(own, other, attacked, position_flags)) == [(8<<56, 2<<56)]

    # for move in king_castle_moves(own, other, attacked, position_flags):
    #     print('From:')
    #     print_bb(move[0])
    #     print('To:')
    #     print_bb(move[1])
    #     print()

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
