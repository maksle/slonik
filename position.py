from piece_type import *
from move_gen import *
from bb import *
from move import *
import tt
import math
import itertools

Pt = PieceType

class Position():
    def __init__(self, pos = None):
        if pos is not None:
            self.position_flags = pos.position_flags

            self.squares = pos.squares[:]
            self.pieces = pos.pieces[:]
            self.occupied = pos.occupied[:]
            self.moves = pos.moves[:]
            self.pinned = pos.pinned[:]
            self.sliding_checkers = pos.sliding_checkers[:]
            self.discoverers = pos.discoverers[:]
            self.k_lines = pos.k_lines[:]
            
            self.w_king_move_ply = pos.w_king_move_ply
            self.w_king_castle_ply = pos.w_king_castle_ply
            self.w_kr_move_ply = pos.w_kr_move_ply
            self.w_qr_move_ply = pos.w_qr_move_ply
            self.b_king_move_ply = pos.b_king_move_ply
            self.b_king_castle_ply = pos.b_king_castle_ply
            self.b_kr_move_ply = pos.b_kr_move_ply
            self.b_qr_move_ply = pos.b_qr_move_ply

            self.en_pessant_sq = -1

            self.zobrist_hash = pos.zobrist_hash

        else:
            self.position_flags = Side.WHITE << 6

            self.w_king_move_ply = -1
            self.w_king_castle_ply = -1
            self.w_kr_move_ply = -1
            self.w_qr_move_ply = -1
            self.b_king_move_ply = -1
            self.b_king_castle_ply = -1
            self.b_kr_move_ply = -1
            self.b_qr_move_ply = -1

            self.en_pessant_sq = -1
            
            self.init_squares()
            self.init_pieces()
            self.init_occupied()
            self.init_zobrist()
            
            self.moves = []

            # filled during evaluation
            self.pinned = [0 for i in range(13)]
            self.discoverers = [0 for i in range(13)]
            self.sliding_checkers = [0 for i in range(13)]

            # updated only when king moves
            self.load_king_lines()

    def load_king_lines(self):
        wk = self.pieces[Pt.piece(Pt.K, Side.WHITE)]
        bk = self.pieces[Pt.piece(Pt.K, Side.BLACK)]
        # occ = self.occupied[Side.WHITE] | self.occupied[Side.BLACK]
        self.k_lines = [queen_attack(wk, 0), queen_attack(bk, 0)]
            
    def init_squares(self):
        self.squares = [PieceType.NULL for i in range(0,64)]
        self.squares[0] = self.squares[7] = PieceType.W_ROOK
        self.squares[1] = self.squares[6] = PieceType.W_KNIGHT
        self.squares[2] = self.squares[5] = PieceType.W_BISHOP
        self.squares[3] = PieceType.W_KING
        self.squares[4] = PieceType.W_QUEEN
        self.squares[56] = self.squares[63] = PieceType.B_ROOK
        self.squares[57] = self.squares[62] = PieceType.B_KNIGHT
        self.squares[58] = self.squares[61] = PieceType.B_BISHOP
        self.squares[59] = PieceType.B_KING
        self.squares[60] = PieceType.B_QUEEN

    def init_pieces(self):
        self.pieces = [None] * 13
        self.pieces[PieceType.NULL] = 0

        self.pieces[PieceType.W_PAWN] = 0xff00
        self.pieces[PieceType.W_KNIGHT] = 0x42
        self.pieces[PieceType.W_BISHOP] = 0x24
        self.pieces[PieceType.W_QUEEN] = 0x10
        self.pieces[PieceType.W_KING] = 0x8
        self.pieces[PieceType.W_ROOK] = 0x81

        self.pieces[PieceType.B_PAWN] = 0xff00 << 40
        self.pieces[PieceType.B_KNIGHT] = 0x42 << 56
        self.pieces[PieceType.B_BISHOP] = 0x24 << 56
        self.pieces[PieceType.B_QUEEN] = 0x10 << 56
        self.pieces[PieceType.B_KING] = 0x8 << 56
        self.pieces[PieceType.B_ROOK] = 0x81 << 56

    def init_occupied(self):
        self.occupied = [self.get_occupied(Side.WHITE),
                         self.get_occupied(Side.BLACK)]

    def init_zobrist(self):
        self.zobrist_hash = 0

        # pieces
        for piece_type, pieces in enumerate(self.pieces):
            if piece_type != PieceType.NULL:
                self.zobrist_hash ^= zobrist_pieces(pieces, piece_type)

        # side to move
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[self.side_to_move()]

        # castling
        if self.w_kr_move_ply == -1 and self.w_king_move_ply == -1:
            self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
        if self.w_qr_move_ply == -1 and self.w_king_move_ply == -1:
            self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]
        if self.b_kr_move_ply == -1 and self.b_king_move_ply == -1:
            self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
        if self.b_qr_move_ply == -1 and self.b_king_move_ply == -1:
            self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]

    @classmethod
    def from_fen(cls, fen):
        pieces, color, castling, en_pessant, halfmove_clock, move_num = fen.split()
        position = Position()

        # side to move
        side = Side.WHITE if color == "w" else Side.BLACK
        position_flags = side << 6

        # pre-init
        position.squares = [PieceType.NULL for i in range(0,64)]
        position.pieces = [0] * 13
        position.pieces[PieceType.NULL] = 0

        # pieces
        pieces_by_row = pieces.split("/")
        for row, pieces in enumerate(reversed(pieces_by_row)):
            col = 0
            for char in reversed(pieces):
                if char.isnumeric():
                    col += int(char)
                else:
                    piece_type = None
                    color = Side.WHITE if char.isupper() else Side.BLACK
                    lchar = char.lower()
                    if lchar == "p":
                        piece_type = PieceType.piece(PieceType.P, color)
                    elif lchar == "n":
                        piece_type = PieceType.piece(PieceType.N, color)
                    elif lchar == "b":
                        piece_type = PieceType.piece(PieceType.B, color)
                    elif lchar == "r":
                        piece_type = PieceType.piece(PieceType.R, color)
                    elif lchar == "q":
                        piece_type = PieceType.piece(PieceType.Q, color)
                    elif lchar == "k":
                        piece_type = PieceType.piece(PieceType.K, color)
                    assert(piece_type is not None)
                    bit_pos = row * 8 + col
                    position.squares[bit_pos] = piece_type
                    position.pieces[piece_type] |= 1 << bit_pos
                    col += 1

        # castling
        if "K" not in castling:
            if "Q" not in castling:
                position_flags |= 1
            else:
                position_flags |= (1 << 2)
        elif "Q" not in castling:
            position_flags |= (1 << 3)

        if "k" not in castling:
            if "q" not in castling:
                position_flags |= 2
            else:
                position_flags |= (1 << 4)
        elif "q" not in castling:
            position_flags |= (1 << 5)

        # en-pessant
        if en_pessant != "-":
            ep_sq = HUMAN_BOARD[en_pessant]
            pawn = PieceType.piece(PieceType.P, side ^ 1)
            if side ^ 1 == Side.WHITE:
                position.moves.append(Move(pawn, ep_sq >> 8, ep_sq << 8))
            else:
                position.moves.append(Move(pawn, ep_sq << 8, ep_sq >> 8))
            position.en_pessant_sq = ep_sq

        # nothing to update for halfmove clock or fullmove num (yet)

        # order matters
        position.position_flags = position_flags
        position.init_occupied()
        position.load_discoveries_and_pins()
        position.init_zobrist()
        
        return position

    def get_occupied(self, side):
        occupied = 0
        for piece_type, piece in enumerate(self.pieces):
            if PieceType.get_side(piece_type) == side:
                occupied |= piece
        return occupied
    
    def generate_moves_all(self):
        us = self.side_to_move()
        valid_sqs = FULL_BOARD
        yield from self.generate_moves_pt(Pt.piece(Pt.P, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.N, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.B, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.R, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.Q, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.K, us), valid_sqs)

    def generate_moves_violent(self):
        us = self.side_to_move()
        them = us ^ 1
        valid_sqs = self.occupied[them]

        # include promotions
        yield from self.generate_moves_pt(Pt.piece(Pt.P, us), valid_sqs, True)

        yield from self.generate_moves_pt(Pt.piece(Pt.N, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.B, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.R, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.Q, us), valid_sqs)
        yield from self.generate_moves_pt(Pt.piece(Pt.K, us), valid_sqs)
        
    def generate_moves_in_check(self):
        us = self.side_to_move()
        them = us ^ 1
        occ = self.occupied[0] | self.occupied[1]
        
        yield from self.generate_moves_pt(Pt.piece(Pt.K, us), invert(us))

        ksq = self.pieces[Pt.piece(Pt.K, us)]
        step_checkers = piece_attack(Pt.N, ksq, occ) & self.pieces[Pt.piece(Pt.N, them)]
        step_checkers |= piece_attack(Pt.piece(Pt.P, us), ksq, occ) & self.pieces[Pt.piece(Pt.P, them)]
        
        sliding_checkers = self.sliding_checkers[Pt.piece(Pt.K, us)]
        if reset_ls1b(sliding_checkers | step_checkers) == 0:
            # there's only one checker, we can try capture or block
            for checker in iterate_pieces(sliding_checkers | step_checkers):
                valid_sqs = checker
                if checker & sliding_checkers:
                    valid_sqs |= BETWEEN_SQS[bit_position(ksq)][bit_position(checker)]
                yield from self.generate_moves_pt(Pt.piece(Pt.P, us), valid_sqs)
                yield from self.generate_moves_pt(Pt.piece(Pt.N, us), valid_sqs)
                yield from self.generate_moves_pt(Pt.piece(Pt.B, us), valid_sqs)
                yield from self.generate_moves_pt(Pt.piece(Pt.R, us), valid_sqs)
                yield from self.generate_moves_pt(Pt.piece(Pt.Q, us), valid_sqs)
    
    def generate_moves_pt(self, pt, valid_sqs=None, do_promo=False):
        if valid_sqs is None:
            valid_sqs = FULL_BOARD

        side = self.side_to_move()
        last_move = self.last_move()
        bt = Pt.base_type(pt)

        own = self.occupied[side]
        other = self.occupied[side ^ 1]
        
        for p in iterate_pieces(self.pieces[pt]):
            if bt == PieceType.P:
                moves = pawn_moves(p, own, other, side, last_move.piece_type, last_move.from_sq, last_move.to_sq)
                # include en-pessant
                if last_move.piece_type != Pt.NULL \
                   and last_move.to_sq & valid_sqs \
                   and shift_north(shift_north(last_move.to_sq, side), side) == last_move.from_sq:
                    valid_sqs |= shift_north(last_move.to_sq, side)
                # include promo
                if do_promo:
                    valid_sqs |= RANKS[0] | RANKS[7]
            elif bt == PieceType.N:
                moves = knight_moves(p, own)
            elif bt == PieceType.B:
                moves = bishop_moves(p, own, other)
            elif bt == PieceType.R:
                moves = rook_moves(p, own, other)
            elif bt == PieceType.Q:
                moves = queen_moves(p, own, other)
            elif bt == PieceType.K:
                moves = pseudo_king_moves(self)
                
            for from_sq, to_sq in moves:
                if to_sq & valid_sqs:
                    is_capture = to_sq & other
                    if bt == Pt.P and to_sq & (RANKS[0] | RANKS[7]):
                        for promo_bt in [Pt.N, Pt.B, Pt.R, Pt.Q]:
                            promo_pt = Pt.piece(promo_bt, side)
                            move = Move(pt, from_sq, to_sq, MoveType.promo, promo_pt)
                            if is_capture: move.move_type |= MoveType.capture
                            yield move
                    else:
                        move = Move(pt, from_sq, to_sq, MoveType.regular)
                        if is_capture: move.move_type |= MoveType.capture
                        yield move
    
    def last_move(self):
        return self.moves[-1] if len(self.moves) else Move(PieceType.NULL, None, None)

    def side_to_move(self):
        return side_to_move(self.position_flags)

    def white_to_move(self):
        return white_to_move(self.position_flags)

    def black_to_move(self):
        return black_to_move(self.position_flags)

    def is_legal(self, move):
        side = self.side_to_move()
        us = side
        them = side ^ 1

        # king moves
        if Pt.base_type(move.piece_type) == Pt.K:
            return self.is_legal_king_move(move)

        # en pessant move
        if Pt.base_type(move.piece_type) == Pt.P \
           and move.to_sq & (shift_ne(move.from_sq, us) | shift_nw(move.from_sq, us)) \
           and self.last_move() == Move(Pt.piece(Pt.P, them), shift_north(move.to_sq, us), shift_south(move.to_sq, us)):
            moved = Position(self)
            moved.make_move(move)
            if moved.in_check():
                return False

        # pinned pieces can only move in line with the king or it's illegal
        if self.pinned[Pt.piece(Pt.K, us)] & move.from_sq \
           and not LINE_SQS[bit_position(self.pieces[Pt.piece(Pt.K, us)])][bit_position(move.from_sq)] & move.to_sq:
            return False
        return True

    def is_legal_king_move(self, move):
        assert Pt.base_type(move.piece_type) == Pt.K
        castle_params = [[E1, G1, 0xe], [E1, C1, 0x70], [E8, G8, 0xe << 56], [E8, C8, 0x70 << 56]]
        for params in castle_params:
            from_sq, to_sq, check_sqs = params
            if move.from_sq == from_sq and move.to_sq == to_sq:
                for sq in iterate_pieces(check_sqs):
                    if self.in_check(Move(move.piece_type, move.from_sq, sq)):
                        return False
                return True
        if self.in_check(move):
            return False
        return True

    def in_check(self, move=None):
        side = self.side_to_move()
        us, them = side, side ^ 1

        sq = self.pieces[Pt.piece(Pt.K, us)]
        occupied = self.occupied[us] | self.occupied[them]

        if move is not None:
            occupied ^= move.from_sq
            occupied |= move.to_sq
            if Pt.base_type(move.piece_type) == Pt.K:
                sq = move.to_sq

        b = knight_attack(sq)
        if b & self.pieces[Pt.piece(Pt.N, them)]:
            return True

        b = pawn_attack(sq, us)
        if b & self.pieces[Pt.piece(Pt.P, them)]:
            return True

        b = bishop_attack(sq, occupied)
        if b & (self.pieces[Pt.piece(Pt.B, them)] | self.pieces[Pt.piece(Pt.Q, them)]):
            return True

        b = rook_attack(sq, occupied)
        if b & (self.pieces[Pt.piece(Pt.R, them)] | self.pieces[Pt.piece(Pt.Q, them)]):
            return True

        b = king_attack(sq)
        if b & self.pieces[Pt.piece(Pt.K, them)]:
            return True

        return False
    
    def get_piece_attacks(self, piece_type, side):
        if side is None:
            side = self.side_to_move()
            piece_type_side = PieceType.piece(piece_type, side)
            occupied = self.occupied[Side.WHITE] | self.occupied[Side.BLACK]
        if piece_type == PieceType.P:
            return pawn_attack(self.pieces[piece_type_side], side)
        elif piece_type == PieceType.N:
            return knight_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.B:
            return bishop_attack(self.pieces[piece_type_side], occupied)
        elif piece_type == PieceType.Q:
            return queen_attack(self.pieces[piece_type_side], occupied)
        elif piece_type == PieceType.K:
            return king_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.R:
            return rook_attack(self.pieces[piece_type_side], occupied)

    def get_attacks(self, side):
        occupied = self.occupied[Side.WHITE] | self.occupied[Side.BLACK]
        if side == Side.WHITE:
            w_pawn_attacks = pawn_attack(self.pieces[PieceType.W_PAWN], side)
            w_knights_attacks = knight_attack(self.pieces[PieceType.W_KNIGHT])
            w_bishops_attacks = bishop_attack(self.pieces[PieceType.W_BISHOP], occupied)
            w_queens_attacks = queen_attack(self.pieces[PieceType.W_QUEEN], occupied)
            w_king_attacks = king_attack(self.pieces[PieceType.W_KING])
            w_rooks_attacks = rook_attack(self.pieces[PieceType.W_ROOK], occupied)
            return w_pawn_attacks \
                | w_knights_attacks \
                | w_bishops_attacks \
                | w_queens_attacks \
                | w_king_attacks \
                | w_rooks_attacks
        else:
            b_pawn_attacks = pawn_attack(self.pieces[PieceType.B_PAWN], side)
            b_knights_attacks = knight_attack(self.pieces[PieceType.B_KNIGHT])
            b_bishops_attacks = bishop_attack(self.pieces[PieceType.B_BISHOP], occupied)
            b_queens_attacks = queen_attack(self.pieces[PieceType.B_QUEEN], occupied)
            b_king_attacks = king_attack(self.pieces[PieceType.B_KING])
            b_rooks_attacks = rook_attack(self.pieces[PieceType.B_ROOK], occupied)
            return b_pawn_attacks \
                | b_knights_attacks \
                | b_bishops_attacks \
                | b_queens_attacks \
                | b_king_attacks \
                | b_rooks_attacks
    
    def make_null_move(self):
        self.moves.append(Move(PieceType.NULL))
        self.toggle_side_to_move()

    def undo_null_move(self):
        assert(self.moves[-1] == Move(PieceType.NULL))
        self.moves.pop()
        self.toggle_side_to_move()

    def toggle_side_to_move(self):
        self.position_flags ^= 1 << 6
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[0]
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[1]


    def load_discoveries_and_pins(self, target_piece_type=Pt.K):
        self.discoverers, self.pinned, self.sliding_checkers = self.get_discoveries_and_pins(target_piece_type)
        
    def get_discoveries_and_pins(self, target_piece_type=Pt.K):
        """Return squares of singular pieces between sliders and the king of side
        `side`. Blockers of opposite side can move and cause discovered check, and
        blockers of same side are pinned. `target_piece_type` is piece things are pinned to."""

        pinned = [0 for i in range(13)]
        discoverers = [0 for i in range(13)]
        sliding_checkers = [0 for i in range(13)]
        
        for side in [Side.WHITE, Side.BLACK]:
            us = side
            them = side ^ 1

            sq = bit_position(self.pieces[Pt.piece(target_piece_type, us)])
            
            if sq and sq > 0:
                occ = self.occupied[us] | self.occupied[them]

                diag_snipers = self.pieces[Pt.piece(Pt.B, them)] | self.pieces[Pt.piece(Pt.Q, them)]
                diag_snipers &= PSEUDO_ATTACKS[PieceType.B][sq]

                line_snipers = self.pieces[Pt.piece(Pt.R, them)] | self.pieces[Pt.piece(Pt.Q, them)]
                line_snipers &= PSEUDO_ATTACKS[PieceType.R][sq]

                for sniper in iterate_pieces(diag_snipers | line_snipers):
                    sniper_sq = bit_position(sniper)
                    b = BETWEEN_SQS[sq][sniper_sq] & occ
                    if b and reset_ls1b(b) == 0:
                        if self.occupied[us] & b:
                            pinned[Pt.piece(target_piece_type, us)] |= b
                        elif self.occupied[them] & b:
                            discoverers[Pt.piece(target_piece_type, us)] |= b
                    elif not b:
                        sliding_checkers[Pt.piece(target_piece_type, us)] |= sniper

        return discoverers, pinned, sliding_checkers

    def make_move(self, move):
        piece_type = move.piece_type
        base_type = PieceType.base_type(piece_type)
        from_sq = move.from_sq
        to_sq = move.to_sq
        side = Side.WHITE if PieceType.is_white(piece_type) else Side.BLACK
        last_move = self.last_move()
        this_move_num = len(self.moves)
        from_square_ind = bit_position(from_sq)
        to_square_ind = bit_position(to_sq)

        capture_mask = to_sq ^ FULL_BOARD
        captured_pt = self.squares[bit_position(to_sq)]
        
        self.position_flags = self.position_flags ^ (1 << 6)

        self.pieces[piece_type] ^= from_sq ^ to_sq
        self.occupied[side] ^= from_sq ^ to_sq

        self.pieces[captured_pt] &= capture_mask
        self.occupied[side ^ 1] &= capture_mask
        
        # zobrist - side
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[0]
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[1]
        
        # zobrist - move's from_sq and move's to_sq
        self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[from_square_ind][piece_type]
        self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][piece_type]
        
        # zobrist - captured piece
        self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][captured_pt]
        
        # en pessant
        en_pessant_capture = False
        if base_type == Pt.P:
            pawn_them = Pt.piece(Pt.P, side ^ 1)
            if last_move.piece_type == pawn_them \
               and last_move.from_sq == shift_north(to_sq, side) \
               and last_move.to_sq == shift_south(to_sq, side):
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[bit_position(last_move.to_sq)][pawn_them]
                self.pieces[pawn_them] ^= last_move.to_sq
                self.occupied[side ^ 1] ^= last_move.to_sq
                self.squares[bit_position(last_move.to_sq)] = PieceType.NULL
                self.en_pessant_sq = -1
                en_pessant_capture = True
            if shift_south(shift_south(from_sq, side), side) == to_sq:
                self.en_pessant_sq = shift_south(from_sq, side)

        # castling
        if base_type == Pt.K:
            if side == Side.WHITE:
                self.w_king_move_ply = this_move_num
                self.position_flags |= 1
                if self.w_kr_move_ply == -1 and self.w_king_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
                elif self.w_qr_move_ply == -1 and self.w_king_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]

                if from_sq == E1 and to_sq == G1:
                    self.w_kr_move_ply = this_move_num
                    self.w_king_castle_ply = this_move_num
                    self.position_flags |= 4
                    self.occupied[Side.WHITE] ^= 0x5
                    self.pieces[PieceType.W_ROOK] ^= 0x5
                    self.squares[bit_position(H1)] = PieceType.NULL
                    self.squares[bit_position(F1)] = PieceType.W_ROOK
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[0][PieceType.W_ROOK]
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[2][PieceType.W_ROOK]
                elif from_sq == E1 and to_sq == C1:
                    self.w_qr_move_ply = this_move_num
                    self.w_king_castle_ply = this_move_num
                    self.position_flags |= 8
                    self.occupied[Side.WHITE] ^= 0x90
                    self.pieces[PieceType.W_ROOK] ^= 0x90
                    self.squares[bit_position(A1)] = PieceType.NULL
                    self.squares[bit_position(D1)] = PieceType.W_ROOK
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[7][PieceType.W_ROOK]
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[4][PieceType.W_ROOK]
            else: # Black
                self.b_king_move_ply = this_move_num
                self.position_flags |= 2
                if self.b_kr_move_ply == -1 and self.b_king_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
                elif self.b_qr_move_ply == -1 and self.b_king_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]

                if from_sq == E8 and to_sq == G8:
                    self.position_flags |= 16
                    self.b_kr_move_ply = this_move_num
                    self.b_king_castle_ply = this_move_num
                    self.occupied[Side.BLACK] ^= (0x5 << 56)
                    self.pieces[PieceType.B_ROOK] ^= (0x5 << 56)
                    self.squares[bit_position(H8)] = PieceType.NULL
                    self.squares[bit_position(F8)] = PieceType.B_ROOK
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[56][PieceType.B_ROOK]
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[58][PieceType.B_ROOK]
                elif from_sq == E8 and to_sq == C8:
                    self.position_flags |= 32
                    self.b_qr_move_ply = this_move_num
                    self.b_king_castle_ply = this_move_num
                    self.occupied[Side.BLACK] ^= (0x90 << 56)
                    self.pieces[PieceType.B_ROOK] ^= (0x90 << 56)
                    self.squares[bit_position(A8)] = PieceType.NULL
                    self.squares[bit_position(D8)] = PieceType.B_ROOK
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[63][PieceType.B_ROOK]
                    self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[60][PieceType.B_ROOK]
        
        # castling rights
        if base_type == Pt.R:
            if side == Side.WHITE:
                if from_sq == H1 and self.w_kr_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
                    self.position_flags |= 4
                    self.w_kr_move_ply = this_move_num
                elif from_sq == A1 and self.w_qr_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]
                    self.position_flags |= 8
                    self.w_qr_move_ply = this_move_num
            else: # Black
                if from_sq == H8 and self.b_kr_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
                    self.position_flags |= 16
                    self.b_kr_move_ply = this_move_num
                elif from_sq == A8 and self.b_qr_move_ply == -1:
                    self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]
                    self.position_flags |= 32
                    self.b_qr_move_ply = this_move_num
        
        # promotions
        if base_type == PieceType.P and get_rank(to_sq, side) == 7:
            self.pieces[piece_type] ^= to_sq
            self.pieces[move.promo_piece] ^= to_sq
            self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][piece_type]
            self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][move.promo_piece]
        
        if base_type == Pt.K:
            self.load_king_lines()

        if (move.from_sq | move.to_sq) & (self.k_lines[side] | self.k_lines[side^1]) or en_pessant_capture:
            self.load_discoveries_and_pins(target_piece_type=Pt.K)

        self.squares[bit_position(from_sq)] = PieceType.NULL
        self.squares[bit_position(to_sq)] = piece_type if not move.move_type == MoveType.promo else move.promo_piece

        self.moves.append(move)

def zobrist_pieces(pieces, piece_type):
    zobrist_hash = 0
    for piece in iterate_pieces(pieces):
        square_ind = bit_position(piece)
        zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[square_ind][piece_type]
    return zobrist_hash
