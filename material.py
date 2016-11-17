from bb import *

MG_PIECES = [None for i in range(0, 7)]
MG_PIECES[PieceType.P.value] = 200
MG_PIECES[PieceType.N.value] = 650
MG_PIECES[PieceType.B.value] = 650
MG_PIECES[PieceType.R.value] = 1000
MG_PIECES[PieceType.Q.value] = 1950
MG_PIECES[PieceType.K.value] = 0

def material_eval(counts, piece_t, side):
    if piece_t == PieceType.P.value:
        return pawn_value(counts, side)
    elif piece_t == PieceType.N.value:
        return knight_value(counts, side)
    elif piece_t == PieceType.B.value:
        return bishop_value(counts, side)
    elif piece_t == PieceType.R.value:
        return rook_value(counts, side)
    elif piece_t == PieceType.Q.value:
        return queen_value(counts, side)

def piece_counts(pos):
    counts = [None for pt in range(0, 13)]
    side = pos.side_to_move()
    piece_types = [PieceType.P, PieceType.N, PieceType.B, PieceType.R, PieceType.Q]
    sides = [Side.WHITE.value, Side.BLACK.value]
    for pt in piece_types:
        for s in sides:
            spt = PieceType.piece(pt.value, s)
            counts[spt] = count_bits(pos.pieces[spt])
    return counts

def knight_value(counts, side):
    piece_type = PieceType.N.value
    spt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[spt] == 0:
        return 0

    val = MG_PIECES[piece_type]

    # knight gains value with more pawns on board, and loses value with less
    # pawns on board
    p_us_cnt = counts[PieceType.piece(PieceType.P.value, us)]
    p_them_cnt = counts[PieceType.piece(PieceType.P.value, them)]
    untraded_p_pairs = (p_them_cnt + p_us_cnt) / 2
    val += (untraded_p_pairs - 5) * .0625 * MG_PIECES[PieceType.P.value]

    # very slight redundancy with another knight
    n_us_cnt = counts[spt]
    if n_us_cnt > 1:
        val -= .02 * MG_PIECES[PieceType.P.value]
    # print("knight:", val)
    return val

def bishop_value(counts, side):
    piece_type = PieceType.B.value
    us_pt = PieceType.piece(piece_type, side)
    us, them = side, side ^ 1

    if counts[us_pt] == 0:
        return 0

    val = MG_PIECES[piece_type]

    # bishop pair bonus
    b_us_cnt = counts[us_pt]
    if b_us_cnt > 1:
        val += .5 * MG_PIECES[PieceType.P.value]
        # even more when no other minors
        other_b = counts[PieceType.piece(PieceType.B.value, them)]
        other_n = counts[PieceType.piece(PieceType.N.value, them)]
        if other_b + other_n == 0:
            val += .25 * MG_PIECES[PieceType.P.value]
    # print("bishop:", val)
    return val

def rook_value(counts, side):
    piece_type = PieceType.R.value
    us, them = side, side ^ 1
    r_us_cnt = counts[PieceType.piece(piece_type, us)]

    if r_us_cnt == 0:
        return 0

    val = MG_PIECES[piece_type]

    # redundancy penalty, encourage rook trades
    q_us_cnt = counts[PieceType.piece(piece_type, us)]
    q_them_cnt = counts[PieceType.piece(PieceType.Q.value, them)]
    # in imbalance, the side without the rook favors keeping rooks
    if not q_us_cnt < q_them_cnt:
        val -= (r_us_cnt - 1) * .04 * MG_PIECES[PieceType.P.value]
    if q_us_cnt > 0:
        val -= .02 * MG_PIECES[PieceType.P.value]

    # rook loses value per additional pawn each side, gains value with more
    # pawns off board
    p_us_cnt = counts[PieceType.piece(PieceType.P.value, us)]
    p_them_cnt = counts[PieceType.piece(PieceType.P.value, them)]
    untraded_p_pairs = (p_us_cnt + p_them_cnt) / 2
    val -= untraded_p_pairs * .0125 * MG_PIECES[PieceType.P.value]
    # print("rook:", val)
    return val

def queen_value(counts, side):
    piece_type = PieceType.Q.value
    us, them = side, side ^ 1
    q_us_cnt = counts[PieceType.piece(piece_type, us)]

    if q_us_cnt == 0:
        return 0

    val = MG_PIECES[piece_type]

    n_us_cnt = counts[PieceType.piece(PieceType.N.value, us)]
    b_us_cnt = counts[PieceType.piece(PieceType.B.value, us)]
    if n_us_cnt + b_us_cnt >= 2:
        val += .25 * MG_PIECES[PieceType.P.value]
    # print("queen:", val)
    return val

def pawn_value(counts, side):
    return MG_PIECES[PieceType.P.value]
