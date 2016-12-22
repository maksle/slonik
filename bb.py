"""Parallel prefix routines (kogge-stone) for calculating ray attacks, and other
piece attacks"""

from print_bb import *
from side import Side
from piece_type import PieceType
import gmpy2

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

# bottom up 0-7
RANKS = [0xff << (8*i) for i in range(8)]
# A==0, H==7
FILES = [0x0101010101010101 << (i) for i in range(7,-1,-1)]

WHITE_SQUARES = 0xAA55AA55AA55AA55
DARK_SQUARES = WHITE_SQUARES ^ FULL_BOARD

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

def rook_attack_calc(g, p):
    return north_attack(g, p) \
        | east_attack(g, p) \
        | south_attack(g, p) \
        | west_attack(g, p)

def queen_attack_calc(g, p):
    return rook_attack_calc(g, p) | bishop_attack_calc(g, p)

def bishop_attack_calc(g, p):
    return nw_attack(g, p) \
        | ne_attack(g, p) \
        | se_attack(g, p) \
        | sw_attack(g, p)

def knight_attack_calc(g):
    attacks = ((g << 6) & NOT_A_FILE & NOT_B_FILE) \
        | ((g >> 10) & NOT_A_FILE & NOT_B_FILE) \
        | ((g >> 17) & NOT_A_FILE) \
        | ((g >> 15) & NOT_H_FILE) \
        | ((g >> 6) & NOT_G_FILE & NOT_H_FILE) \
        | ((g << 10) & NOT_G_FILE & NOT_H_FILE) \
        | ((g << 15) & NOT_A_FILE) \
        | ((g << 17) & NOT_H_FILE)
    return attacks & ((1 << 64) - 1)

def king_attack_calc(g):
    return ((g << 9) & NOT_H_FILE) \
        | g << 8 \
        | ((g << 7) & NOT_A_FILE) \
        | ((g << 1) & NOT_H_FILE) \
        | ((g >> 1) & NOT_A_FILE) \
        | ((g >> 7) & NOT_H_FILE) \
        | g >> 8 \
        | ((g >> 9) & NOT_A_FILE)

def pawn_attack_calc(pawn, side_to_move):
    if side_to_move == Side.WHITE:
        return ((pawn << 9) & NOT_H_FILE) \
            | ((pawn << 7) & NOT_A_FILE)
    else:
        return ((pawn >> 9) & NOT_A_FILE) \
            | ((pawn >> 7) & NOT_H_FILE)

def piece_attack(pt, sq, occupied):
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

# def count_bits(b):
#     n = 0
#     while b > 0:
#         n = n + 1
#         b = reset_ls1b(b)
#     return n

# def count_bits(b):
#     return bin(b).count("1")

def count_bits(b):
    return gmpy2.popcount(b)

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

def invert(b):
    return b ^ FULL_BOARD

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

def bit_position(square):
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

def rook_mask(sq):
    attacks = PSEUDO_ATTACKS[PieceType.R][sq]
    edges = edge_mask(sq)
    return attacks & (FULL_BOARD ^ edges)

def bishop_mask(sq):
    attacks = PSEUDO_ATTACKS[PieceType.B][sq]
    edges = edge_mask(sq)
    return attacks & (FULL_BOARD ^ edges)

    
BETWEEN_SQS = [[0] * 65 for i in range(64)]
LINE_SQS = [[0] * 65 for i in range(64)]

PSEUDO_ATTACKS = [[0] * 64 for i in range(7)]
MAGIC_MASKS = [[0] * 64 for i in range(7)]
for sq_ind in range(64):
    for pt in [PieceType.B, PieceType.R]:
        sq = 1 << sq_ind

        attack_fn = bishop_attack_calc if pt == PieceType.B else rook_attack_calc
        attacks = attack_fn(sq, FULL_BOARD)
        PSEUDO_ATTACKS[pt][sq_ind] = attacks
        PSEUDO_ATTACKS[PieceType.Q][sq_ind] |= attacks

        mask_fn = bishop_mask if pt == PieceType.B else rook_mask
        mask = mask_fn(sq_ind)
        MAGIC_MASKS[pt][sq_ind] = mask
        MAGIC_MASKS[PieceType.Q][sq_ind] |= mask

for sq_ind in range(64):
    for sq2_ind in range(64):
        for pt in [PieceType.B, PieceType.R]:
            attack_fn = bishop_attack_calc if pt == PieceType.B else rook_attack_calc
            sq = 1 << sq_ind
            sq2 = 1 << sq2_ind
            if PSEUDO_ATTACKS[pt][sq_ind] & sq2 == 0:
                continue
            BETWEEN_SQS[sq_ind][sq2_ind] = attack_fn(sq, sq2 ^ FULL_BOARD) & attack_fn(sq2, sq ^ FULL_BOARD)
            LINE_SQS[sq_ind][sq2_ind] = (attack_fn(sq, FULL_BOARD) & attack_fn(sq2, FULL_BOARD)) | sq | sq2

AHEAD_SQS = [[0 for i in range(64)] for s in range(2)]
for sq_ind in range(64):
    for side in [Side.WHITE, Side.BLACK]:
        last_rank = 7 if side == Side.WHITE else 0
        f = 7 - (sq_ind % 8)
        last_sq = FILES[f] & RANKS[last_rank]
        AHEAD_SQS[side][sq_ind] = BETWEEN_SQS[sq_ind][bit_position(last_sq)] | last_sq

ATTACKS = [[0] * 64 for i in range(7)]
for pt in [Pt.N, Pt.K]:
    for sq in range(64):
        if pt == Pt.N: ATTACKS[pt][sq] = knight_attack_calc(1 << sq)
        elif pt == Pt.K: ATTACKS[pt][sq] = king_attack_calc(1 << sq)

def knight_attack(sq):
    return ATTACKS[Pt.N][bit_position(sq)]

def king_attack(sq):
    return ATTACKS[Pt.K][bit_position(sq)]

def pawn_attack(sq, side):
    return pawn_attack_calc(sq, side)

def bishop_attack(sq, occupied):
    bitpos = bit_position(sq)
    occ = occupied & MAGIC_MASKS[PieceType.B][bitpos]
    occ *= MAGIC_NUMBER[PieceType.B][bitpos]
    occ &= FULL_BOARD
    hash_index = occ >> (64 - MASK_BIT_LENGTH[PieceType.B][bitpos])
    return MAGIC_ATTACKS[PieceType.B][bitpos][hash_index]

def rook_attack(sq, occupied):
    bitpos = bit_position(sq)
    occ = occupied & MAGIC_MASKS[PieceType.R][bitpos]
    occ *= MAGIC_NUMBER[PieceType.R][bitpos]
    occ &= FULL_BOARD
    hash_index = occ >> (64 - MASK_BIT_LENGTH[PieceType.R][bitpos])
    return MAGIC_ATTACKS[PieceType.R][bitpos][hash_index]

def queen_attack(sq, occupied):
    return bishop_attack(sq, occupied) | rook_attack(sq, occupied)

MASK_BIT_LENGTH = [[] for i in range(7)]
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

MAGIC_ATTACKS = [[] for i in range(7)]
MAGIC_ATTACKS[PieceType.B] = [[None] * 512 for i in range(64)]
MAGIC_ATTACKS[PieceType.R] = [[None] * 4096 for i in range(64)]

def index_to_occupation(index, bits, mask):
    m = mask
    result = 0
    for i in range(bits):
        j = ls1b(m)
        m = reset_ls1b(m)
        if index & (1 << i):
            result |= j
    return result
            
MAGIC_NUMBER = [[] for i in range(7)]
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

print("Initializing Magics")
for pt in [PieceType.B, PieceType.R]:
    for sq, magic in enumerate(MAGIC_NUMBER[pt]):
        mask = MAGIC_MASKS[pt][sq]
        bits = MASK_BIT_LENGTH[pt][sq]
        for index in range(1 << bits):
            occupation = index_to_occupation(index, bits, mask)
            free = invert(occupation) # calc funcs take free, not occupied
            index_hash = ((occupation * magic) & FULL_BOARD) >> (64 - bits)
            attack_fn = bishop_attack_calc if pt == PieceType.B else rook_attack_calc
            MAGIC_ATTACKS[pt][sq][index_hash] = attack_fn(1 << sq, free)
