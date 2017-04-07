# cython: profile=False

"""Parallel prefix routines (kogge-stone) for calculating ray attacks, and other
piece attacks"""

from print_bb import *
from side import Side
from piece_type import PieceType

import gmpy2
cimport numpy as np
import numpy as np
from numpy import uint64
from numpy cimport uint64_t

Pt = PieceType

A_FILE = 0x8080808080808080
B_FILE = 0x4040404040404040
G_FILE = 0x0202020202020202
H_FILE = 0x0101010101010101
FULL_BOARD = 0xffffffffffffffff

NOT_A_FILE = A_FILE ^ FULL_BOARD
NOT_H_FILE = H_FILE ^ FULL_BOARD
NOT_B_FILE = B_FILE ^ FULL_BOARD
NOT_G_FILE = G_FILE ^ FULL_BOARD


ctypedef unsigned long long ULL
cdef:
    ULL _AFILE, _B_FILE, _G_FILE, _H_FILE
    ULL _NOT_A_FILE, _NOT_B_FILE, _NOT_G_FILE, _NOT_H_FILE
    ULL _FULL_BOARD
_A_FILE = 0x8080808080808080
_B_FILE = 0x4040404040404040
_G_FILE = 0x0202020202020202
_H_FILE = 0x0101010101010101
_FULL_BOARD = 0xffffffffffffffff
_NOT_A_FILE = A_FILE ^ FULL_BOARD
_NOT_H_FILE = H_FILE ^ FULL_BOARD
_NOT_B_FILE = B_FILE ^ FULL_BOARD
_NOT_G_FILE = G_FILE ^ FULL_BOARD


# bottom up 0-7
RANKS = [0xff << (8*i) for i in range(8)]
# A==0, H==7
FILES = [0x0101010101010101 << (i) for i in range(7,-1,-1)]

WHITE_SQUARES = 0xAA55AA55AA55AA55
DARK_SQUARES = WHITE_SQUARES ^ FULL_BOARD

HUMAN_PIECE = {}
HUMAN_PIECE[PieceType.NULL] = ""
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

PICTURE_PIECES = {
    PieceType.NULL: '·',
    PieceType.PAWN: '♙',
    PieceType.KNIGHT: '♘',
    PieceType.BISHOP: '♗',
    PieceType.ROOK: '♖',
    PieceType.QUEEN: '♕',
    PieceType.KING: '♔',
    PieceType.B_PAWN: '♟',
    PieceType.B_KNIGHT: '♞',
    PieceType.B_BISHOP: '♝',
    PieceType.B_ROOK: '♜',
    PieceType.B_QUEEN: '♛',
    PieceType.B_KING: '♚',
}
    
cpdef ULL north(ULL b):
    return b << 8

cpdef ULL south(ULL b):
    return b >> 8

cdef ULL east(ULL b):
    return b >> 1 & _NOT_A_FILE

cpdef ULL west(ULL b):
    return b << 1 & _NOT_H_FILE

cpdef ULL north_east(ULL b):
    return b << 7 & _NOT_A_FILE

cpdef ULL north_west(ULL b):
    return b << 9 & _NOT_H_FILE & FULL_BOARD

cpdef ULL south_east(ULL b):
    return b >> 9 & _NOT_A_FILE

cpdef ULL south_west(ULL b):
    return b >> 7 & _NOT_H_FILE

cpdef ULL west_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0
    cdef ULL pr1
    cdef ULL pr2
    pr0 = p & _NOT_H_FILE
    pr1 = pr0 & (pr0 << 1)
    pr2 = pr1 & (pr1 << 2)
    g |= pr0 & g << 1
    g |= pr1 & g << 2
    g |= pr2 & g << 4
    return (g << 1) & _NOT_H_FILE

cpdef ULL east_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0
    cdef ULL pr1
    cdef ULL pr2
    pr0 = p & _NOT_A_FILE
    pr1 = pr0 & (pr0 >> 1)
    pr2 = pr1 & (pr1 >> 2)
    g |= pr0 & g >> 1
    g |= pr1 & g >> 2
    g |= pr2 & g >> 4
    return (g >> 1) & _NOT_A_FILE

cpdef ULL north_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    g |= p & (g << 8)
    p &= (p << 8)
    g |= p & (g << 16)
    p &= (p << 16)
    g |= p & (g << 32)
    return (g << 8)

cpdef ULL south_attack(ULL g, ULL p):
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

cpdef ULL ne_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0
    cdef ULL pr1
    cdef ULL pr2
    pr0 = p & _NOT_A_FILE
    pr1 = pr0 & (pr0 << 7)
    pr2 = pr1 & (pr1 << 14)
    g |= pr0 & g << 7
    g |= pr1 & g << 14
    g |= pr2 & g << 28
    return (g << 7) & _NOT_A_FILE

cpdef nw_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0
    cdef ULL pr1
    cdef ULL pr2
    pr0 = p & _NOT_H_FILE
    pr1 = pr0 & (pr0 << 9)
    pr2 = pr1 & (pr1 << 18)
    g |= pr0 & g << 9
    g |= pr1 & g << 18
    g |= pr2 & g << 36
    return (g << 9) & _NOT_H_FILE

cpdef ULL se_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0
    cdef ULL pr1
    cdef ULL pr2
    pr0 = p & _NOT_A_FILE
    pr1 = pr0 & (pr0 >> 9)
    pr2 = pr1 & (pr1 >> 18)
    g |= pr0 & g >> 9
    g |= pr1 & g >> 18
    g |= pr2 & g >> 36
    return g >> 9 & _NOT_A_FILE

cpdef ULL sw_attack(ULL g, ULL p):
    """ (g: int, p: int) -> int
    g: attacker
    p: free squares
    """
    cdef ULL pr0, pr1, pr2
    pr0 = p & _NOT_H_FILE
    pr1 = pr0 & (pr0 >> 7)
    pr2 = pr1 & (pr1 >> 14)
    g |= pr0 & g >> 7
    g |= pr1 & g >> 14
    g |= pr2 & g >> 28
    return g >> 7 & _NOT_H_FILE

cpdef ULL rook_attack_calc(ULL g, ULL p):
    return north_attack(g, p) \
        | east_attack(g, p) \
        | south_attack(g, p) \
        | west_attack(g, p)

cpdef ULL queen_attack_calc(ULL g, ULL p):
    return rook_attack_calc(g, p) | bishop_attack_calc(g, p)

cpdef ULL bishop_attack_calc(ULL g, ULL p):
    return nw_attack(g, p) \
        | ne_attack(g, p) \
        | se_attack(g, p) \
        | sw_attack(g, p)

cpdef ULL knight_attack_calc(ULL g):
    cdef ULL attacks
    attacks = ((g << 6) & _NOT_A_FILE & _NOT_B_FILE) \
        | ((g >> 10) & _NOT_A_FILE & _NOT_B_FILE) \
        | ((g >> 17) & _NOT_A_FILE) \
        | ((g >> 15) & _NOT_H_FILE) \
        | ((g >> 6) & _NOT_G_FILE & _NOT_H_FILE) \
        | ((g << 10) & _NOT_G_FILE & _NOT_H_FILE) \
        | ((g << 15) & _NOT_A_FILE) \
        | ((g << 17) & _NOT_H_FILE)
    return attacks

cpdef ULL king_attack_calc(ULL g):
    return ((g << 9) & _NOT_H_FILE) \
        | g << 8 \
        | ((g << 7) & _NOT_A_FILE) \
        | ((g << 1) & _NOT_H_FILE) \
        | ((g >> 1) & _NOT_A_FILE) \
        | ((g >> 7) & _NOT_H_FILE) \
        | g >> 8 \
        | ((g >> 9) & _NOT_A_FILE)

cpdef ULL pawn_attack_calc(ULL pawn, int side_to_move):
    if side_to_move == Side.WHITE:
        return ((pawn << 9) & _NOT_H_FILE) \
            | ((pawn << 7) & _NOT_A_FILE)
    else:
        return ((pawn >> 9) & _NOT_A_FILE) \
            | ((pawn >> 7) & _NOT_H_FILE)

cpdef piece_attack(int pt, ULL sq, ULL occupied):
    cdef int bt
    bt = Pt.base_type(pt)
    if bt == Pt.P:
        return pawn_attack(sq, Pt.get_side(pt))
    elif bt == Pt.N:
        return knight_attack(sq)
    elif bt == Pt.B:
        return bishop_attack(sq, occupied)
    elif bt == Pt.R:
        return rook_attack(sq, occupied)
    elif bt == Pt.Q:
        return bishop_attack(sq, occupied) | rook_attack(sq, occupied)
    elif bt == Pt.K:
        return king_attack(sq)
    
cpdef ULL exclude_own(ULL attacks, ULL own):
    cdef ULL not_own
    not_own = own ^ _FULL_BOARD
    return attacks & not_own

cpdef ULL ls1b(ULL p):
    # Least significant 1 bit
    return p & -p

cpdef ULL reset_ls1b(ULL p):
    # flip least significant 1 bit
    return p & (p-1)

def iterate_pieces(ULL b):
    cdef ULL board
    board = b
    while board > 0:
        yield ls1b(board)
        board = reset_ls1b(board)

# def count_bits(b):
#     n = 0
#     while b > 0:
#         n = n + 1
#         b = reset_ls1b(b)
#     return n

# def count_bits(b):
#     return bin(b).count("1")

cpdef ULL count_bits(ULL b):
    return gmpy2.popcount(b)

cpdef ULL invert(ULL b):
    return b ^ _FULL_BOARD

# de_bruijn_bitpos = [
#     63,  0, 58,  1, 59, 47, 53,  2,
#     60, 39, 48, 27, 54, 33, 42,  3,
#     61, 51, 37, 40, 49, 18, 28, 20,
#     55, 30, 34, 11, 43, 14, 22,  4,
#     62, 57, 46, 52, 38, 26, 32, 41,
#     50, 36, 17, 19, 29, 10, 13, 21,
#     56, 45, 25, 31, 35, 16,  9, 12,
#     44, 24, 15,  8, 23,  7,  6,  5
# ]
# def bit_position(square):
#     return de_bruijn_bitpos[((ls1b(square) * 0x07edd5e59a4e28c2) & FULL_BOARD) >> 58]

# surprisingly this is faster than using the de bruijn sequence
# def bit_position(square):
#     return len(bin(square))-3

cpdef bit_position(ULL square):
    return gmpy2.bit_scan1(square)

def shift_north(n, side):
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

def get_rank(n, side=None):
    rank = bit_position(n) // 8
    if side is None or side == Side.WHITE:
        return rank
    else:
        return 7 - rank


# MAGIC BITBOARDS helpers
def edge_mask(sq):
    edges = 0
    rk = sq // 8
    fl = 7 - (sq % 8)
    if not rk == 0: edges |= RANKS[0]
    if not rk == 7: edges |= RANKS[7]
    if not fl == 0: edges |= FILES[0]
    if not fl == 7: edges |= FILES[7]
    return edges

cdef ULL rook_mask(int sq):
    cdef ULL attacks, edges
    cdef int ptr
    ptr = PieceType.R
    attacks = PSEUDO_ATTACKS[ptr, sq]
    edges = edge_mask(sq)
    return attacks & (_FULL_BOARD ^ edges)

cdef ULL bishop_mask(int sq):
    cdef ULL attacks, edges
    cdef int ptb
    ptb = PieceType.B
    attacks = PSEUDO_ATTACKS[ptb][sq]
    edges = edge_mask(sq)
    return attacks & (_FULL_BOARD ^ edges)

    
# BETWEEN_SQS = [[0] * 65 for i in range(64)]
BETWEEN_SQS = np.zeros((64, 65), dtype='Q')
# LINE_SQS = [[0] * 65 for i in range(64)]
LINE_SQS = np.zeros((64, 65), dtype='Q')

# PSEUDO_ATTACKS = [[0] * 64 for i in range(7)]
PSEUDO_ATTACKS = np.zeros((7, 64), dtype='Q')
# MAGIC_MASKS = [[0] * 64 for i in range(7)]
MAGIC_MASKS = np.zeros((7, 64), dtype='Q')
# AHEAD_SQS = [[0 for i in range(64)] for s in range(2)]
AHEAD_SQS = np.zeros((2, 64), dtype='Q')
# ATTACKS = [[0] * 64 for i in range(7)]
ATTACKS = np.zeros((7, 64), dtype='Q')
cdef:
    int pt, qpt, side, last_rank, sq_orig, sq2_orig
    ULL sq, sq2, attacks, mask, magic
    ULL[:,:] BETWEEN_SQS_V = BETWEEN_SQS
    ULL[:,:] LINE_SQS_V = LINE_SQS
    ULL[:,:] PSEUDO_ATTACKS_V = PSEUDO_ATTACKS
    ULL[:,:] MAGIC_MASKS_V = MAGIC_MASKS
    ULL[:,:] AHEAD_SQS_V = AHEAD_SQS
    ULL[:,:] ATTACKS_V = ATTACKS
qpt = PieceType.Q
for sq_ind in range(64):
    for pt in [PieceType.B, PieceType.R]:
        sq = 1 << sq_ind
        sq_orig = sq_ind
        
        attack_fn = bishop_attack_calc if pt == PieceType.B else rook_attack_calc
        attacks = attack_fn(sq, FULL_BOARD)
        PSEUDO_ATTACKS_V[pt, sq_orig] = attacks
        PSEUDO_ATTACKS_V[qpt, sq_orig] |= attacks

        mask_fn = bishop_mask if pt == PieceType.B else rook_mask
        mask = mask_fn(sq_ind)
        MAGIC_MASKS_V[pt, sq_orig] = mask
        MAGIC_MASKS_V[qpt, sq_orig] |= mask

for sq_ind in range(64):
    for sq2_ind in range(64):
        for pt in [PieceType.B, PieceType.R]:
            attack_fn = bishop_attack_calc if pt == PieceType.B else rook_attack_calc
            sq = 1 << sq_ind
            sq2 = 1 << sq2_ind
            sq_orig = sq_ind
            sq2_orig = sq2_ind
            if PSEUDO_ATTACKS_V[pt, sq_orig] & sq2 == 0:
                continue
            BETWEEN_SQS_V[sq_orig, sq2_orig] = attack_fn(sq, sq2 ^ _FULL_BOARD) & attack_fn(sq2, sq ^ _FULL_BOARD)
            LINE_SQS_V[sq_orig, sq2_orig] = (attack_fn(sq, FULL_BOARD) & attack_fn(sq2, FULL_BOARD)) | sq | sq2

cdef int bp_last_sq
for sq_ind in range(64):
    for side in [Side.WHITE, Side.BLACK]:
        last_rank = 7 if side == Side.WHITE else 0
        f = 7 - (sq_ind % 8)
        last_sq = FILES[f] & RANKS[last_rank]
        sq_orig = sq_ind
        bp_last_sq = bit_position(last_sq)
        AHEAD_SQS_V[side, sq_orig] = BETWEEN_SQS_V[sq_orig, bp_last_sq] | last_sq

for pt in [Pt.N, Pt.K]:
    for sq_ind in range(64):
        sq_orig = sq_ind
        if pt == Pt.N: ATTACKS_V[pt, sq_orig] = knight_attack_calc(1 << sq_ind)
        elif pt == Pt.K: ATTACKS_V[pt, sq_orig] = king_attack_calc(1 << sq_ind)

cpdef ULL pseudo_attacks(int pt, int sq):
    return PSEUDO_ATTACKS_V[pt, sq]

cpdef between_sqs(int sq1, int sq2):
    return BETWEEN_SQS_V[sq1, sq2]

cpdef line_sqs(int sq1, int sq2):
    return LINE_SQS_V[sq1, sq2]

cpdef ULL knight_attack(ULL sq) except -1:
    return ATTACKS[Pt.N][bit_position(sq)]

cpdef ULL king_attack(ULL sq) except -1:
    cdef int ptk
    ptk = Pt.K
    return ATTACKS[ptk, bit_position(sq)]

cpdef ULL pawn_attack(ULL sq, ULL side) except -1:
    return pawn_attack_calc(sq, side)

cpdef ULL bishop_attack(ULL sq, ULL occupied) except -1:
    cdef int ptb, bitpos, hash_index
    cdef ULL occ
    ptb = PieceType.B
    bitpos = bit_position(sq)
    occ = occupied & MAGIC_MASKS_V[ptb, bitpos]
    occ *= MAGIC_NUMBER_V[ptb, bitpos]
    # occ &= FULL_BOARD
    hash_index = occ >> (64 - MASK_BIT_LENGTH_V[ptb, bitpos])
    return MAGIC_ATTACKS_B_V[bitpos, hash_index]

cpdef ULL rook_attack(ULL sq, ULL occupied) except -1:
    cdef int ptr, bitpos, hash_index
    cdef ULL occ
    bitpos = bit_position(sq)
    ptr = PieceType.R
    occ = occupied & MAGIC_MASKS_V[ptr, bitpos]
    occ *= MAGIC_NUMBER_V[ptr, bitpos]
    # occ &= FULL_BOARD
    hash_index = occ >> (64 - MASK_BIT_LENGTH_V[ptr, bitpos])
    return MAGIC_ATTACKS_R_V[bitpos, hash_index]

cpdef ULL queen_attack(ULL sq, ULL occupied) except -1:
    return bishop_attack(sq, occupied) | rook_attack(sq, occupied)

# MASK_BIT_LENGTH = [[] for i in range(7)]
MASK_BIT_LENGTH = np.zeros((7, 64), dtype='int32')
cdef int[:,:] MASK_BIT_LENGTH_V = MASK_BIT_LENGTH
MASK_BIT_LENGTH[PieceType.B] = [
  6, 5, 5, 5, 5, 5, 5, 6,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 7, 7, 7, 7, 5, 5,
  5, 5, 7, 9, 9, 7, 5, 5,
  5, 5, 7, 9, 9, 7, 5, 5,
  5, 5, 7, 7, 7, 7, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  6, 5, 5, 5, 5, 5, 5, 6
];

MASK_BIT_LENGTH[PieceType.R] = [
  12, 11, 11, 11, 11, 11, 11, 12,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  12, 11, 11, 11, 11, 11, 11, 12
];

# MAGIC_ATTACKS = [[] for i in range(7)]
MAGIC_ATTACKS_B = np.zeros((64, 512), dtype='uint64')
MAGIC_ATTACKS_R = np.zeros((64, 4096), dtype='uint64')
# MAGIC_ATTACKS[PieceType.B] = [[None] * 512 for i in range(64)]
# MAGIC_ATTACKS[PieceType.R] = [[None] * 4096 for i in range(64)]
cdef ULL[:,:] MAGIC_ATTACKS_B_V = MAGIC_ATTACKS_B
cdef ULL[:,:] MAGIC_ATTACKS_R_V = MAGIC_ATTACKS_R

cdef ULL index_to_occupation(ULL index, int bits, ULL mask):
    cdef ULL m, j, result
    cdef int i
    m = mask
    result = 0
    for i in range(bits):
        j = ls1b(m)
        m = reset_ls1b(m)
        if index & (1 << i):
            result |= j
    return result
            
# MAGIC_NUMBER = [[] for i in range(7)]
MAGIC_NUMBER = np.zeros((5,64), dtype='ulonglong')
cdef ULL[:,:] MAGIC_NUMBER_V = MAGIC_NUMBER

MAGIC_NUMBER[PieceType.B] = [
    9304441232522551809,
    5197197952615010356,
    290631748237168640,
    1262155822310490112,
    9264203776884064257,
    4684385864763049009,
    2313025036383944704,
    1443478456087938120,
    
    774090679976016,
    1306430928654044196,
    9223522687463227393,
    153764521465348226,
    3458768981727394832,
    4791866292238811168,
    18019449408331792,
    594476269954934304,
    
    649675308061426176,
    9225626087636239376,
    9225624437976732160,
    2348631614491214848,
    4612820716585222404,
    1163336218600539908,
    4611967502002823172,
    4611971893606649872,
    
    4522497571819584,
    11819710416710353408,
    110340390449709568,
    586594127551402240,
    3476924047873318916,
    144258128899737088,
    1173198708800522768,
    3171099286652911910,
    
    1154082898125521664,
    4612814125981503490,
    21392253016736776,
    167160127293952,
    4917934366771970304,
    4748069518046184193,
    563517258269696,
    1171227274402468352,
    
    466228367853600,
    1153486723179741744,
    334354681235464,
    37154980429432832,
    9259436156364260608,
    6757632831390208,
    3794354570994385412,
    9224506751188523264,
    
    74810278297600,
    76708530406555656,
    5800637426016723456,
    18172728729665601,
    2267776679936,
    70403120890912,
    90090701441875968,
    2603661418834214928,
    
    4641542200153342988,
    73184600445159570,
    653021948120926336,
    703687443908608,
    110762613381923332,
    4665738048838314020,
    3553759087100480,
    582759550845696
]

MAGIC_NUMBER[PieceType.R] = [
    9295448606603477248,
    9241387810290683904,
    10520418350834692225,
    72092797872636160,
    4755810075845592096,
    144119594713360417,
    288241373550806032,
    16573247456075124868,

    2460232036091172897,
    1196286367813636,
    74450682706399232,
    140806208356480,
    5764889067082770432,
    5765170799975600136,
    562984346723353,
    1459729231436079362,

    4629842803698974720,
    4612289651384976258,
    1481684827431239808,
    364792119866427392,
    504544445576775680,
    9261793921032258560,
    55173493501081616,
    1161196429135478852,

    2322170705379616,
    35185445851136,
    2312633599472435330,
    563233421811728,
    15024012757110360064,
    4901042846389370884,
    31527482315244872,
    10151799449209988,

    105828136780165,
    864726313921941504,
    2324210488411427073,
    4611695914585690116,
    2310351013338417152,
    577023738864734340,
    649659673972050192,
    4611688922932576516,

    4646811287846914,
    76562616910168064,
    9007341006553152,
    4900479551766921242,
    585542752725434384,
    2486057380769956112,
    9223427012503339264,
    1134981632360452,

    7068963674293174784,
    1171287747375138880,
    13511366354670720,
    671040845612404864,
    720858892842566912,
    283676147712128,
    4684307116537156096,
    180322144682496,

    19804136227362,
    282420943265921,
    141021226801666,
    3459046126504641929,
    576742846327162881,
    9289233718052865,
    864832974180356108,
    145294587333762
]

cdef int bits, index_hash
cdef ULL free, occupation
print("Initializing Magics")
for pt in [PieceType.B, PieceType.R]:
    for sq_ind, magic in enumerate(MAGIC_NUMBER[pt]):
        sq_orig = sq_ind
        mask = MAGIC_MASKS_V[pt, sq_orig]
        bits = MASK_BIT_LENGTH_V[pt, sq_orig]
        for index in range(1 << bits):
            occupation = index_to_occupation(index, bits, mask)
            free = invert(occupation) # calc funcs take free, not occupied
            index_hash = (occupation * magic) >> (64 - bits)
            if pt == Pt.B:
                MAGIC_ATTACKS_B_V[sq_orig, index_hash] = bishop_attack_calc(1 << sq_ind, free)
            else:
                MAGIC_ATTACKS_R_V[sq_orig, index_hash] = rook_attack_calc(1 << sq_ind, free)
