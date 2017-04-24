from bb import *
from math import tanh

MG_PIECES = [None for i in range(0, 7)]
MG_PIECES[PieceType.NULL] = 0
MG_PIECES[PieceType.P] = 188
MG_PIECES[PieceType.N] = 753
MG_PIECES[PieceType.B] = 826
MG_PIECES[PieceType.R] = 1285
MG_PIECES[PieceType.Q] = 2513
MG_PIECES[PieceType.K] = 10000

EG_PIECES = [None for i in range(0, 7)]
EG_PIECES[PieceType.NULL] = 0
EG_PIECES[PieceType.P] = 248
EG_PIECES[PieceType.N] = 832
EG_PIECES[PieceType.B] = 897
EG_PIECES[PieceType.R] = 1371
EG_PIECES[PieceType.Q] = 2650
EG_PIECES[PieceType.K] = 10000

PHASE_WEIGHT = [None for i in range(0, 7)]
PHASE_WEIGHT[PieceType.P] = 0
PHASE_WEIGHT[PieceType.N] = 1
PHASE_WEIGHT[PieceType.B] = 1
PHASE_WEIGHT[PieceType.R] = 2
PHASE_WEIGHT[PieceType.Q] = 4

MAX_PHASE = PHASE_WEIGHT[Pt.P] * 16 \
            + PHASE_WEIGHT[Pt.N] * 4 \
            + PHASE_WEIGHT[Pt.B] * 4 \
            + PHASE_WEIGHT[Pt.R] * 4 \
            + PHASE_WEIGHT[Pt.Q] * 2
    
def get_phase(pos):
    pcs = [PieceType.W_PAWN, PieceType.W_KNIGHT, PieceType.W_BISHOP,
           PieceType.W_ROOK, PieceType.W_QUEEN,
           PieceType.B_PAWN, PieceType.B_KNIGHT, PieceType.B_BISHOP,
           PieceType.B_ROOK, PieceType.B_QUEEN]
    return sum([
        count_bits(pos.pieces[pc]) * PHASE_WEIGHT[PieceType.base_type(pc)] for pc in pcs
    ])
    
def scale_phase(mg, eg, phase):
    # return mg
    delta = mg - eg
    return eg + delta * phase / MAX_PHASE

def material_bootstrap(pos):
    wp_count = count_bits(pos.pieces[Pt.P])
    wn_count = count_bits(pos.pieces[Pt.N])
    wb_count = count_bits(pos.pieces[Pt.B])
    wr_count = count_bits(pos.pieces[Pt.R])
    wq_count = count_bits(pos.pieces[Pt.Q])
    bp_count = count_bits(pos.pieces[Pt.BP])
    bn_count = count_bits(pos.pieces[Pt.BN])
    bb_count = count_bits(pos.pieces[Pt.BB])
    br_count = count_bits(pos.pieces[Pt.BR])
    bq_count = count_bits(pos.pieces[Pt.BQ])

    phase = wp_count * PHASE_WEIGHT[Pt.P] \
            + wn_count * PHASE_WEIGHT[Pt.N] \
            + wb_count * PHASE_WEIGHT[Pt.B] \
            + wr_count * PHASE_WEIGHT[Pt.R] \
            + wq_count * PHASE_WEIGHT[Pt.Q]
    phase += bp_count * PHASE_WEIGHT[Pt.P] \
            + bn_count * PHASE_WEIGHT[Pt.N] \
            + bb_count * PHASE_WEIGHT[Pt.B] \
            + br_count * PHASE_WEIGHT[Pt.R] \
            + bq_count * PHASE_WEIGHT[Pt.Q]

    if phase > MAX_PHASE:
        phase = MAX_PHASE

    ret = wp_count * scale_phase(MG_PIECES[Pt.P], EG_PIECES[Pt.P], phase)
    ret += wn_count * scale_phase(MG_PIECES[Pt.N], EG_PIECES[Pt.N], phase)
    ret += wb_count * scale_phase(MG_PIECES[Pt.B], EG_PIECES[Pt.B], phase)
    ret += wr_count * scale_phase(MG_PIECES[Pt.R], EG_PIECES[Pt.R], phase)
    ret += wq_count * scale_phase(MG_PIECES[Pt.Q], EG_PIECES[Pt.Q], phase)

    ret -= bp_count * scale_phase(MG_PIECES[Pt.P], EG_PIECES[Pt.P], phase)
    ret -= bn_count * scale_phase(MG_PIECES[Pt.N], EG_PIECES[Pt.N], phase)
    ret -= bb_count * scale_phase(MG_PIECES[Pt.B], EG_PIECES[Pt.B], phase)
    ret -= br_count * scale_phase(MG_PIECES[Pt.R], EG_PIECES[Pt.R], phase)
    ret -= bq_count * scale_phase(MG_PIECES[Pt.Q], EG_PIECES[Pt.Q], phase)

    return 1000 * tanh(1e-3 * ret)
    
def material_eval(phase, counts, piece_t, side):
    if piece_t == PieceType.P:
        return pawn_value(phase, counts, side)
    elif piece_t == PieceType.N:
        return knight_value(phase, counts, side)
    elif piece_t == PieceType.B:
        return bishop_value(phase, counts, side)
    elif piece_t == PieceType.R:
        return rook_value(phase, counts, side)
    elif piece_t == PieceType.Q:
        return queen_value(phase, counts, side)

def piece_counts(pos):
    counts = [None for pt in range(0, 13)]
    side = pos.side_to_move()
    piece_types = [PieceType.P, PieceType.N, PieceType.B, PieceType.R, PieceType.Q]
    sides = [Side.WHITE, Side.BLACK]
    for pt in piece_types:
        for s in sides:
            spt = PieceType.piece(pt, s)
            counts[spt] = count_bits(pos.pieces[spt])
    return counts

def knight_value(phase, counts, side):
    piece_type = PieceType.N
    spt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[spt] == 0:
        return 0

    # val = MG_PIECES[piece_type]
    val = scale_phase(MG_PIECES[piece_type], EG_PIECES[piece_type], phase)

    # knight gains with more pawns on board, and loses with less
    # pawns on board
    p_us_cnt = counts[PieceType.piece(PieceType.P, us)]
    p_them_cnt = counts[PieceType.piece(PieceType.P, them)]
    untraded_p_pairs = (p_them_cnt + p_us_cnt) / 2
    val += (untraded_p_pairs - 5) * .0625 * MG_PIECES[PieceType.P]

    # very slight redundancy with another knight
    n_us_cnt = counts[spt]
    if n_us_cnt > 1:
        val -= .02 * MG_PIECES[PieceType.P]
    # print("knight:", val)
    return val

def bishop_value(phase, counts, side):
    piece_type = PieceType.B
    us_pt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[us_pt] == 0:
        return 0

    # val = MG_PIECES[piece_type]
    val = scale_phase(MG_PIECES[piece_type], EG_PIECES[piece_type], phase)

    # bishop pair bonus
    b_us_cnt = counts[us_pt]
    if b_us_cnt > 1:
        val += .5 * MG_PIECES[PieceType.P]
        # even more when no other minors
        other_b = counts[PieceType.piece(PieceType.B, them)]
        other_n = counts[PieceType.piece(PieceType.N, them)]
        if other_b + other_n == 0:
            val += .25 * MG_PIECES[PieceType.P]
    # print("bishop:", val)
    return val

def rook_value(phase, counts, side):
    piece_type = PieceType.R
    us, them = side, side ^ 1
    r_us_cnt = counts[PieceType.piece(piece_type, us)]

    if r_us_cnt == 0:
        return 0

    val = scale_phase(MG_PIECES[piece_type], EG_PIECES[piece_type], phase)
    # val = MG_PIECES[piece_type]

    # redundancy penalty, encourage rook trades
    q_us_cnt = counts[PieceType.piece(piece_type, us)]
    q_them_cnt = counts[PieceType.piece(PieceType.Q, them)]
    # in imbalance, the side without the rook favors keeping rooks
    if not q_us_cnt < q_them_cnt:
        val -= (r_us_cnt - 1) * .04 * MG_PIECES[PieceType.P]
    if q_us_cnt > 0:
        val -= .02 * MG_PIECES[PieceType.P]

    # rook loses value per additional pawn each side, gains value with more
    # pawns off board
    p_us_cnt = counts[PieceType.piece(PieceType.P, us)]
    p_them_cnt = counts[PieceType.piece(PieceType.P, them)]
    untraded_p_pairs = (p_us_cnt + p_them_cnt) / 2
    val -= untraded_p_pairs * .0125 * MG_PIECES[PieceType.P]
    # print("rook:", val)
    return val

def queen_value(phase, counts, side):
    piece_type = PieceType.Q
    us, them = side, side ^ 1
    q_us_cnt = counts[PieceType.piece(piece_type, us)]

    if q_us_cnt == 0:
        return 0

    # val = MG_PIECES[piece_type]
    val = scale_phase(MG_PIECES[piece_type], EG_PIECES[piece_type], phase)

    n_us_cnt = counts[PieceType.piece(PieceType.N, us)]
    b_us_cnt = counts[PieceType.piece(PieceType.B, us)]
    if n_us_cnt + b_us_cnt >= 2:
        val += .25 * MG_PIECES[PieceType.P]
    # print("queen:", val)
    return val

def pawn_value(phase, counts, side):
    piece_type = Pt.P
    val = scale_phase(MG_PIECES[piece_type], EG_PIECES[piece_type], phase)
    # return MG_PIECES[PieceType.P]
    return val
