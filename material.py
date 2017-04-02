from bb import *

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

def material_eval(counts, piece_t, side):
    if piece_t == PieceType.P:
        return pawn_value(counts, side)
    elif piece_t == PieceType.N:
        return knight_value(counts, side)
    elif piece_t == PieceType.B:
        return bishop_value(counts, side)
    elif piece_t == PieceType.R:
        return rook_value(counts, side)
    elif piece_t == PieceType.Q:
        return queen_value(counts, side)

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

def knight_value(counts, side):
    piece_type = PieceType.N
    spt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[spt] == 0:
        return 0

    val = MG_PIECES[piece_type]

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

def bishop_value(counts, side):
    piece_type = PieceType.B
    us_pt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[us_pt] == 0:
        return 0

    val = MG_PIECES[piece_type]

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

def rook_value(counts, side):
    piece_type = PieceType.R
    us, them = side, side ^ 1
    r_us_cnt = counts[PieceType.piece(piece_type, us)]

    if r_us_cnt == 0:
        return 0

    val = MG_PIECES[piece_type]

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

def queen_value(counts, side):
    piece_type = PieceType.Q
    us, them = side, side ^ 1
    q_us_cnt = counts[PieceType.piece(piece_type, us)]

    if q_us_cnt == 0:
        return 0

    val = MG_PIECES[piece_type]

    n_us_cnt = counts[PieceType.piece(PieceType.N, us)]
    b_us_cnt = counts[PieceType.piece(PieceType.B, us)]
    if n_us_cnt + b_us_cnt >= 2:
        val += .25 * MG_PIECES[PieceType.P]
    # print("queen:", val)
    return val

def pawn_value(counts, side):
    return MG_PIECES[PieceType.P]
