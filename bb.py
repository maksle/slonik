"""Parallel prefix routines (kogge-stone) for calculating ray attacks, and other
piece attacks"""

from print_bb import *
from side import Side
from piece_type import PieceType

A_FILE = 0x8080808080808080
B_FILE = 0x4040404040404040
G_FILE = 0x0202020202020202
H_FILE = 0x0101010101010101
FULL_BOARD = 0xffffffffffffffff

NOT_A_FILE = A_FILE ^ FULL_BOARD
NOT_H_FILE = H_FILE ^ FULL_BOARD
NOT_B_FILE = B_FILE ^ FULL_BOARD
NOT_G_FILE = G_FILE ^ FULL_BOARD

# bottom up 0-7
RANKS = [0xff << (8*i) for i in range(8)]
# A==0, H==7
FILES = [0x0101010101010101 << (i) for i in range(7,-1,-1)]

HUMAN_PIECE = {}
HUMAN_PIECE[PieceType.W_PAWN] = ""
HUMAN_PIECE[PieceType.W_KNIGHT] = "N"
HUMAN_PIECE[PieceType.W_BISHOP] = "B"
HUMAN_PIECE[PieceType.W_ROOK] = "R"
HUMAN_PIECE[PieceType.W_QUEEN] = "Q"
HUMAN_PIECE[PieceType.W_KING] = "K"
HUMAN_PIECE[PieceType.B_PAWN] = ""
HUMAN_PIECE[PieceType.B_KNIGHT] = "N"
HUMAN_PIECE[PieceType.B_BISHOP] = "B"
HUMAN_PIECE[PieceType.B_ROOK] = "R"
HUMAN_PIECE[PieceType.B_QUEEN] = "Q"
HUMAN_PIECE[PieceType.B_KING] = "K"
HUMAN_PIECE_INV = {v: k for k, v in HUMAN_PIECE.items()}

HUMAN_BOARD = {}
SQS = []

A1 = HUMAN_BOARD["a1"] = 0x80
B1 = HUMAN_BOARD["b1"] = 0x40
C1 = HUMAN_BOARD["c1"] = 0x20
D1 = HUMAN_BOARD["d1"] = 0x10
E1 = HUMAN_BOARD["e1"] = 0x8
F1 = HUMAN_BOARD["f1"] = 0x4
G1 = HUMAN_BOARD["g1"] = 2
H1 = HUMAN_BOARD["h1"] = 1

A2 = HUMAN_BOARD["a2"] = 0x80 << 8
B2 = HUMAN_BOARD["b2"] = 0x40 << 8
C2 = HUMAN_BOARD["c2"] = 0x20 << 8
D2 = HUMAN_BOARD["d2"] = 0x10 << 8
E2 = HUMAN_BOARD["e2"] = 0x8 << 8
F2 = HUMAN_BOARD["f2"] = 0x4 << 8
G2 = HUMAN_BOARD["g2"] = 2 << 8
H2 = HUMAN_BOARD["h2"] = 1 << 8

A3 = HUMAN_BOARD["a3"] = 0x80 << 16
B3 = HUMAN_BOARD["b3"] = 0x40 << 16
C3 = HUMAN_BOARD["c3"] = 0x20 << 16
D3 = HUMAN_BOARD["d3"] = 0x10 << 16
E3 = HUMAN_BOARD["e3"] = 0x8 << 16
F3 = HUMAN_BOARD["f3"] = 0x4 << 16
G3 = HUMAN_BOARD["g3"] = 2 << 16
H3 = HUMAN_BOARD["h3"] = 1 << 16

A4 = HUMAN_BOARD["a4"] = 0x80 << 24
B4 = HUMAN_BOARD["b4"] = 0x40 << 24
C4 = HUMAN_BOARD["c4"] = 0x20 << 24
D4 = HUMAN_BOARD["d4"] = 0x10 << 24
E4 = HUMAN_BOARD["e4"] = 0x8 << 24
F4 = HUMAN_BOARD["f4"] = 0x4 << 24
G4 = HUMAN_BOARD["g4"] = 2 << 24
H4 = HUMAN_BOARD["h4"] = 1 << 24

A5 = HUMAN_BOARD["a5"] = 0x80 << 32
B5 = HUMAN_BOARD["b5"] = 0x40 << 32
C5 = HUMAN_BOARD["c5"] = 0x20 << 32
D5 = HUMAN_BOARD["d5"] = 0x10 << 32
E5 = HUMAN_BOARD["e5"] = 0x8 << 32
F5 = HUMAN_BOARD["f5"] = 0x4 << 32
G5 = HUMAN_BOARD["g5"] = 2 << 32
H5 = HUMAN_BOARD["h5"] = 1 << 32

A6 = HUMAN_BOARD["a6"] = 0x80 << 40
B6 = HUMAN_BOARD["b6"] = 0x40 << 40
C6 = HUMAN_BOARD["c6"] = 0x20 << 40
D6 = HUMAN_BOARD["d6"] = 0x10 << 40
E6 = HUMAN_BOARD["e6"] = 0x8 << 40
F6 = HUMAN_BOARD["f6"] = 0x4 << 40
G6 = HUMAN_BOARD["g6"] = 2 << 40
H6 = HUMAN_BOARD["h6"] = 1 << 40

A7 = HUMAN_BOARD["a7"] = 0x80 << 48
B7 = HUMAN_BOARD["b7"] = 0x40 << 48
C7 = HUMAN_BOARD["c7"] = 0x20 << 48
D7 = HUMAN_BOARD["d7"] = 0x10 << 48
E7 = HUMAN_BOARD["e7"] = 0x8 << 48
F7 = HUMAN_BOARD["f7"] = 0x4 << 48
G7 = HUMAN_BOARD["g7"] = 2 << 48
H7 = HUMAN_BOARD["h7"] = 1 << 48

A8 = HUMAN_BOARD["a8"] = 0x80 << 56
B8 = HUMAN_BOARD["b8"] = 0x40 << 56
C8 = HUMAN_BOARD["c8"] = 0x20 << 56
D8 = HUMAN_BOARD["d8"] = 0x10 << 56
E8 = HUMAN_BOARD["e8"] = 0x8 << 56
F8 = HUMAN_BOARD["f8"] = 0x4 << 56
G8 = HUMAN_BOARD["g8"] = 2 << 56
H8 = HUMAN_BOARD["h8"] = 1 << 56

HUMAN_BOARD_INV = {v: k for k, v in HUMAN_BOARD.items()}

SQS = [1<<x for x in range(64)]

# BETWEEN_SQS = [[0] * 64 for i in range(64)]
# offset = 1
# c = -1
# while offset < 65:
#     y = 1 << offset
#     c += 1
#     BETWEEN_SQS[offset][y] = c
#     BETWEEN_SQS[y][offset] = c
    
def north(b):
    return b << 8

def south(b):
    return b >> 8

def east(b):
    return b >> 1 & NOT_A_FILE

def west(b):
    return b << 1 & NOT_H_FILE

def north_east(b):
    return b << 7 & NOT_A_FILE

def north_west(b):
    return b << 9 & NOT_H_FILE & FULL_BOARD

def south_east(b):
    return b >> 9 & NOT_A_FILE

def south_west(b):
    return b >> 7 & NOT_H_FILE

def west_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_H_FILE
    pr1 = pr0 & (pr0 << 1)
    pr2 = pr1 & (pr1 << 2)
    g |= pr0 & g << 1
    g |= pr1 & g << 2
    g |= pr2 & g << 4
    return (g << 1) & NOT_H_FILE & ((1 << 64) - 1)

def east_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_A_FILE
    pr1 = pr0 & (pr0 >> 1)
    pr2 = pr1 & (pr1 >> 2)
    g |= pr0 & g >> 1
    g |= pr1 & g >> 2
    g |= pr2 & g >> 4
    return (g >> 1) & NOT_A_FILE

def north_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    g |= p & (g << 8)
    p &= (p << 8)
    g |= p & (g << 16)
    p &= (p << 16)
    g |= p & (g << 32)
    return (g << 8) & ((1 << 64) - 1)

def south_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    g |= p & (g >> 8)
    p &= p >> 8
    g |= p & (g >> 16)
    p &= p >> 16
    g |= p & (g >> 32)
    return g >> 8

def ne_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_A_FILE
    pr1 = pr0 & (pr0 << 7)
    pr2 = pr1 & (pr1 << 14)
    g |= pr0 & g << 7
    g |= pr1 & g << 14
    g |= pr2 & g << 28
    return (g << 7) & NOT_A_FILE & ((1 << 64) - 1)

def nw_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_H_FILE
    pr1 = pr0 & (pr0 << 9)
    pr2 = pr1 & (pr1 << 18)
    g |= pr0 & g << 9
    g |= pr1 & g << 18
    g |= pr2 & g << 36
    return (g << 9) & NOT_H_FILE & ((1 << 64) - 1)

def se_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_A_FILE
    pr1 = pr0 & (pr0 >> 9)
    pr2 = pr1 & (pr1 >> 18)
    g |= pr0 & g >> 9
    g |= pr1 & g >> 18
    g |= pr2 & g >> 36
    return g >> 9 & NOT_A_FILE

def sw_attack(g, p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    pr0 = p & NOT_H_FILE
    pr1 = pr0 & (pr0 >> 7)
    pr2 = pr1 & (pr1 >> 14)
    g |= pr0 & g >> 7
    g |= pr1 & g >> 14
    g |= pr2 & g >> 28
    return g >> 7 & NOT_H_FILE

def rook_attack(g, p):
    return north_attack(g, p) \
        | east_attack(g, p) \
        | south_attack(g, p) \
        | west_attack(g, p)

def queen_attack(g, p):
    return rook_attack(g, p) | bishop_attack(g, p)

def bishop_attack(g, p):
    return nw_attack(g, p) \
        | ne_attack(g, p) \
        | se_attack(g, p) \
        | sw_attack(g, p)

def knight_attack(g):
    attacks = ((g << 6) & NOT_A_FILE & NOT_B_FILE) \
        | ((g >> 10) & NOT_A_FILE & NOT_B_FILE) \
        | ((g >> 17) & NOT_A_FILE) \
        | ((g >> 15) & NOT_H_FILE) \
        | ((g >> 6) & NOT_G_FILE & NOT_H_FILE) \
        | ((g << 10) & NOT_G_FILE & NOT_H_FILE) \
        | ((g << 15) & NOT_A_FILE) \
        | ((g << 17) & NOT_H_FILE)
    return attacks & ((1 << 64) - 1)

def king_attack(g):
    return ((g << 9) & NOT_H_FILE) \
        | g << 8 \
        | ((g << 7) & NOT_A_FILE) \
        | ((g << 1) & NOT_H_FILE) \
        | ((g >> 1) & NOT_A_FILE) \
        | ((g >> 7) & NOT_H_FILE) \
        | g >> 8 \
        | ((g >> 9) & NOT_A_FILE)

def pawn_attack(pawn, side_to_move):
    if side_to_move == Side.WHITE:
        return ((pawn << 9) & NOT_H_FILE) \
            | ((pawn << 7) & NOT_A_FILE)
    else:
        return ((pawn >> 9) & NOT_A_FILE) \
            | ((pawn >> 7) & NOT_H_FILE)

def exclude_own(attacks, own):
    not_own = own ^ FULL_BOARD
    return attacks & not_own

def ls1b(p):
    # Least significant 1 bit
    return p & -p

def reset_ls1b(p):
    # flip least significant 1 bit
    return p & (p-1)

def iterate_pieces(b):
    board = b
    while board > 0:
        yield ls1b(board)
        board = reset_ls1b(board)

def count_bits(b):
    n = 0
    while b > 0:
        n = n + 1
        b = reset_ls1b(b)
    return n
    
def mask(b, mask_val):
    return b & (mask_val ^ FULL_BOARD)

def en_pessant_sq(side_to_move, last_move_piece, from_sq, to_sq):
    if side_to_move == Side.WHITE \
       and last_move_piece == PieceType.B_PAWN \
       and from_sq & 0xff000000000000 > 0 \
       and to_sq & 0xff00000000 > 0:
        return to_sq << 8
    if side_to_move == Side.BLACK \
       and last_move_piece == PieceType.W_PAWN \
       and from_sq & 0xff00 > 0 \
       and to_sq & 0xff000000 > 0:
        return from_sq << 8
    return 0

def bit_position(square):
    return len(bin(square))-3

def shift_north(n, side, times):
    if side == Side.WHITE:
        return north(n)
    else:
        return south(n)

def shift_south(n, side):
    if side == Side.WHITE:
        return south(n)
    else:
        return north(n)

def shift_east(n, side):
    if side == Side.WHITE:
        return east(n)
    else:
        return west(n)

def shift_west(n, side):
    if side == Side.WHITE:
        return west(n)
    else:
        return east(n)

def shift_ne(n, side):
    if side == Side.WHITE:
        return north_east(n)
    else:
        return south_west(n)   

def shift_nw(n, side):
    if side == Side.WHITE:
        return north_west(n)
    else:
        return south_east(n)

def shift_se(n, side):
    if side == Side.WHITE:
        return south_east(n)
    else:
        return north_west(n)

def shift_sw(n, side):
    if side == Side.WHITE:
        return south_west(n)
    else:
        return north_east(n)

def get_file(n):
    return 7 - (bit_position(n) % 8)

def get_rank(n):
    return bit_position(n) // 8

BETWEEN_SQS = [[0] * 65 for i in range(64)]
LINE_SQS = [[0] * 65 for i in range(64)]

PSEUDO_ATTACKS = [[0] * 64 for i in range(7)]
for sq_ind in range(64):
    for pt in [PieceType.B, PieceType.R]:
        sq = 1 << sq_ind
        attack_fn = bishop_attack if pt == PieceType.B else rook_attack
        PSEUDO_ATTACKS[pt][sq_ind] = attack_fn(sq, FULL_BOARD)
        PSEUDO_ATTACKS[PieceType.Q][sq_ind] |= attack_fn(sq, FULL_BOARD)

for sq_ind in range(64):
    for sq2_ind in range(64):
        for pt in [PieceType.B, PieceType.R]:
            attack_fn = bishop_attack if pt == PieceType.B else rook_attack
            sq = 1 << sq_ind
            sq2 = 1 << sq2_ind
            if PSEUDO_ATTACKS[pt][sq_ind] & sq2 == 0:
                continue
            BETWEEN_SQS[sq_ind][sq2_ind] = attack_fn(sq, sq2 ^ FULL_BOARD) & attack_fn(sq2, sq ^ FULL_BOARD)
            LINE_SQS[sq_ind][sq2_ind] = (attack_fn(sq, FULL_BOARD) & attack_fn(sq2, FULL_BOARD)) | sq | sq2

