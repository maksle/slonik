from piece_type import *
from move_gen import *
from bb import *
from move import *
import tt
import math
import itertools

class Position():
    def __init__(self, pos = None):
        if pos is not None:
            self.position_flags = pos.position_flags

            self.squares = pos.squares[:]
            self.pieces = pos.pieces[:]
            self.occupied = pos.occupied[:]
            self.piece_attacks = pos.piece_attacks[:]
            self.attacks = pos.attacks[:]
            self.moves = pos.moves[:]

            self.w_king_move_ply = pos.w_king_move_ply
            self.w_king_castle_ply = pos.w_king_castle_ply
            self.w_kr_move_ply = pos.w_kr_move_ply
            self.w_qr_move_ply = pos.w_qr_move_ply
            self.b_king_move_ply = pos.b_king_move_ply
            self.b_king_castle_ply = pos.b_king_castle_ply
            self.b_kr_move_ply = pos.b_kr_move_ply
            self.b_qr_move_ply = pos.b_qr_move_ply

            self.zobrist_hash = pos.zobrist_hash

        else:
            self.position_flags = Side.WHITE << 6

            self.init_squares() 
            self.init_pieces()
            self.init_occupied()
            self.init_attacks()
            self.init_zobrist()

            self.moves = []

            self.w_king_move_ply = -1
            self.w_king_castle_ply = -1
            self.w_kr_move_ply = -1
            self.w_qr_move_ply = -1
            self.b_king_move_ply = -1
            self.b_king_castle_ply = -1
            self.b_kr_move_ply = -1
            self.b_qr_move_ply = -1
            
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

    def init_attacks(self):
        self.attacks = [self.get_attacks(Side.WHITE),
                        self.get_attacks(Side.BLACK)]
        self.piece_attacks = [None] * 13
        for base_pt in [PieceType.P, PieceType.N, PieceType.B, PieceType.R, PieceType.Q, PieceType.K]:
            white = PieceType.piece(base_pt, Side.WHITE)
            black = PieceType.piece(base_pt, Side.BLACK)
            self.piece_attacks[white] = self.get_piece_attacks(base_pt, Side.WHITE)
            self.piece_attacks[black] = self.get_piece_attacks(base_pt, Side.BLACK)

    def init_zobrist(self):
        self.zobrist_hash = 0
        for piece_type, pieces in enumerate(self.pieces):
            if piece_type != PieceType.NULL:
                self.zobrist_hash ^= zobrist_pieces(pieces, piece_type)
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[Side.WHITE]
        for ind, rand in enumerate(tt.ZOBRIST_CASTLE):
            self.zobrist_hash ^= rand
            
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
                    col += int(char) - 1
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

        # nothing to update for halfmove clock or fullmove num (yet)
        
        position.init_occupied()
        position.init_attacks()
        position.init_zobrist()
        
        position.position_flags = position_flags
        return position
        
    def blocking_change(self, move):
        """Returns piece types who's piece attacks need to be recalculated, either
because they are being blocked, or will no longer be blocked, by the move."""
        being_blocked = []
        for pt, attacks in enumerate(self.piece_attacks):
            if PieceType.base_type(pt) not in [PieceType.B, PieceType.R, PieceType.Q]:
                continue
            if attacks & (move.from_sq | move.to_sq):
                being_blocked.append(pt)
        return being_blocked
                
    def get_occupied(self, side):
        occupied = 0
        for piece_type, piece in enumerate(self.pieces):
            if PieceType.get_side(piece_type) == side:
                occupied |= piece
        return occupied

    def get_move_candidates(self):
        side = self.side_to_move()
        last_move = self.last_move()

        own = self.occupied[side]
        other = self.occupied[side ^ 1]
        attacked = self.attacks[side ^ 1]
        
        for bt in PieceType.piece_types(base_only=True):
            pt = PieceType.piece(bt, side)
            pieces = self.pieces[pt]
            if bt == PieceType.P:
                moves = pawn_moves(pieces, own, other, side, last_move.piece_type, last_move.from_sq, last_move.to_sq)
            elif bt == PieceType.N:
                moves = knight_moves(pieces, own)
            elif bt == PieceType.B:
                moves = bishop_moves(pieces, own, other)
            elif bt == PieceType.R:
                moves = rook_moves(pieces, own, other)
            elif bt == PieceType.Q:
                moves = queen_moves(pieces, own, other)
            elif bt == PieceType.K:
                moves = itertools.chain(king_castle_moves(own, other, attacked, self.position_flags),
                                        king_moves(pieces, own, attacked))
            for from_sq, to_sq in moves:
                yield Move(pt, from_sq, to_sq, MoveType.regular)
        
    def last_move(self):
        return self.moves[-1] if len(self.moves) else Move(PieceType.NULL, None, None)

    def side_to_move(self):
        return side_to_move(self.position_flags)

    def white_to_move(self):
        return white_to_move(self.position_flags)

    def black_to_move(self):
        return black_to_move(self.position_flags)

    def in_check(self, side=None):
        if side is None:
            side = self.side_to_move()
        stm_king = PieceType.piece(PieceType.K, side)
        return am_in_check(self.attacks[side ^ 1], self.pieces[stm_king])
    
    def generate_moves(self):
        for move in self.get_move_candidates():
            try_move = Position(self)
            try_move.make_move(move)
            # make sure we didn't put our king in check
            if not try_move.in_check(self.side_to_move()):
                if try_move.in_check():
                    move.move_type = MoveType.check
                move.position = try_move
                yield move
            
    def get_piece_attacks(self, piece_type, side):
        if side is None:
            side = self.side_to_move()
        piece_type_side = PieceType.piece(piece_type, side)
        occupied = self.occupied[Side.WHITE] | self.occupied[Side.BLACK]
        free = occupied ^ FULL_BOARD
        if piece_type == PieceType.P:
            return pawn_attack(self.pieces[piece_type_side], side)
        elif piece_type == PieceType.N:
            return knight_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.B:
            return bishop_attack(self.pieces[piece_type_side], free)
        elif piece_type == PieceType.Q:
            return queen_attack(self.pieces[piece_type_side], free)
        elif piece_type == PieceType.K:
            return king_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.R:
            return rook_attack(self.pieces[piece_type_side], free)

    def get_attacks(self, side):
        occupied = self.occupied[Side.WHITE] | self.occupied[Side.BLACK]
        free = occupied ^ FULL_BOARD
        if side == Side.WHITE:
            w_pawn_attacks = pawn_attack(self.pieces[PieceType.W_PAWN], side)
            w_knights_attacks = knight_attack(self.pieces[PieceType.W_KNIGHT])
            w_bishops_attacks = bishop_attack(self.pieces[PieceType.W_BISHOP], free)
            w_queens_attacks = queen_attack(self.pieces[PieceType.W_QUEEN], free)
            w_king_attacks = king_attack(self.pieces[PieceType.W_KING])
            w_rooks_attacks = rook_attack(self.pieces[PieceType.W_ROOK], free)
            return w_pawn_attacks \
                | w_knights_attacks \
                | w_bishops_attacks \
                | w_queens_attacks \
                | w_king_attacks \
                | w_rooks_attacks
        else:
            b_pawn_attacks = pawn_attack(self.pieces[PieceType.B_PAWN], side)
            b_knights_attacks = knight_attack(self.pieces[PieceType.B_KNIGHT])
            b_bishops_attacks = bishop_attack(self.pieces[PieceType.B_BISHOP], free)
            b_queens_attacks = queen_attack(self.pieces[PieceType.B_QUEEN], free)
            b_king_attacks = king_attack(self.pieces[PieceType.B_KING])
            b_rooks_attacks = rook_attack(self.pieces[PieceType.B_ROOK], free)
            return b_pawn_attacks \
                | b_knights_attacks \
                | b_bishops_attacks \
                | b_queens_attacks \
                | b_king_attacks \
                | b_rooks_attacks

    def is_mate(self):
        if not self.in_check():
            return False
        for move in self.generate_moves():
            try_move = move.position
            if not try_move.in_check(side=self.side_to_move()):
                return False
        return True
        
    def toggle_side_to_move(self):
        self.position_flags ^= 1 << 6
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[0]
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[1]
    
    def make_move(self, move):
        piece_type = move.piece_type
        from_sq = move.from_sq
        to_sq = move.to_sq
        side = Side.WHITE if PieceType.is_white(piece_type) else Side.BLACK
        last_move = self.last_move()
        this_move_num = len(self.moves)
        capture_mask = to_sq ^ FULL_BOARD
        from_square_ind = bit_position(from_sq)
        to_square_ind = bit_position(to_sq)
        piece_ind = piece_type - 1

        self.position_flags = self.position_flags ^ (1 << 6)

        self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[from_square_ind][piece_ind]
        self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][piece_ind]
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[0]
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[1]

        if PieceType.is_white(piece_type):
            self.pieces[PieceType.B_PAWN] &= capture_mask
            self.pieces[PieceType.B_KNIGHT] &= capture_mask
            self.pieces[PieceType.B_BISHOP] &= capture_mask
            self.pieces[PieceType.B_ROOK] &= capture_mask
            self.pieces[PieceType.B_QUEEN] &= capture_mask
            self.occupied[Side.BLACK] &= capture_mask
            self.occupied[Side.WHITE] ^= from_sq ^ to_sq

            if to_sq & self.pieces[PieceType.B_PAWN] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_PAWN]
            elif to_sq & self.pieces[PieceType.B_KNIGHT] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_KNIGHT]
            elif to_sq & self.pieces[PieceType.B_BISHOP] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_BISHOP]
            elif to_sq & self.pieces[PieceType.B_ROOK] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_ROOK]
            elif to_sq & self.pieces[PieceType.B_QUEEN] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_QUEEN]

        elif PieceType.is_black(piece_type):
            self.pieces[PieceType.W_PAWN] &= capture_mask
            self.pieces[PieceType.W_KNIGHT] &= capture_mask
            self.pieces[PieceType.W_BISHOP] &= capture_mask
            self.pieces[PieceType.W_ROOK] &= capture_mask
            self.pieces[PieceType.W_QUEEN] &= capture_mask
            self.occupied[Side.WHITE] &= capture_mask
            self.occupied[Side.BLACK] ^= from_sq ^ to_sq

            if to_sq & self.pieces[PieceType.B_PAWN] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_PAWN]
            elif to_sq & self.pieces[PieceType.B_KNIGHT] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_KNIGHT]
            elif to_sq & self.pieces[PieceType.B_BISHOP] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_BISHOP]
            elif to_sq & self.pieces[PieceType.B_ROOK] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_ROOK]
            elif to_sq & self.pieces[PieceType.B_QUEEN] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_QUEEN]

        if piece_type == PieceType.W_PAWN:
            self.pieces[PieceType.W_PAWN] ^= (from_sq | to_sq)

            if last_move.piece_type == PieceType.B_PAWN \
               and last_move.from_sq == to_sq << 8 \
               and last_move.to_sq == to_sq >> 8:
                self.occupied[Side.BLACK] ^= last_move.to_sq
                self.pieces[PieceType.B_PAWN] ^= last_move.to_sq
                self.squares[bit_position(last_move.to_sq)] = PieceType.NULL

        elif piece_type == PieceType.W_KNIGHT:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_BISHOP:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_QUEEN:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_KING:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

            if self.w_kr_move_ply == -1 and self.w_king_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
            elif self.w_qr_move_ply == -1 and self.w_king_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]

            self.w_king_move_ply = this_move_num
            self.position_flags = self.position_flags | 1

            if from_sq == E1 and to_sq == G1:
                self.w_kr_move_ply = this_move_num
                self.w_king_castle_ply = this_move_num
                self.position_flags = self.position_flags | (1 << 2)

                self.occupied[Side.WHITE] ^= 0x5
                self.pieces[PieceType.W_ROOK] ^= 0x5
                self.squares[bit_position(H1)] = PieceType.NULL
                self.squares[bit_position(F1)] = PieceType.W_ROOK
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[0][PieceType.W_ROOK - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[2][PieceType.W_ROOK - 1]

            elif from_sq == E1 and to_sq == C1:
                self.w_qr_move_ply = this_move_num
                self.w_king_castle_ply = this_move_num
                self.position_flags = self.position_flags | (1 << 3)

                self.occupied[Side.WHITE] ^= 0x90
                self.pieces[PieceType.W_ROOK] ^= 0x90
                self.squares[bit_position(A1)] = PieceType.NULL
                self.squares[bit_position(D1)] = PieceType.W_ROOK
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[7][PieceType.W_ROOK - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[4][PieceType.W_ROOK - 1]

        elif piece_type == PieceType.W_ROOK:
            self.pieces[PieceType.W_ROOK] ^= (from_sq | to_sq)

            if from_sq == H1 and self.w_kr_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
                self.position_flags = self.position_flags | (1 << 2)
                self.w_kr_move_ply = this_move_num
            elif from_sq == A1 and self.w_qr_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]
                self.position_flags = self.position_flags | (1 << 3)
                self.w_qr_move_ply = this_move_num

        elif piece_type == PieceType.B_PAWN:
            self.pieces[PieceType.B_PAWN] ^= (from_sq | to_sq)

            if last_move.piece_type == PieceType.W_PAWN \
               and last_move.from_sq == to_sq >> 8 \
               and last_move.to_sq == to_sq << 8:
                self.occupied[Side.WHITE] ^= last_move.to_sq
                self.pieces[PieceType.W_PAWN] ^= last_move.to_sq
                self.squares[bit_position(last_move.to_sq)] = PieceType.NULL

        elif piece_type == PieceType.B_KNIGHT:
            self.pieces[PieceType.B_KNIGHT] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_BISHOP:
            self.pieces[PieceType.B_BISHOP] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_QUEEN:
            self.pieces[PieceType.B_QUEEN] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_KING:
            self.pieces[PieceType.B_KING] ^= (from_sq | to_sq)

            if self.b_kr_move_ply == -1 and self.b_king_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
            elif self.b_qr_move_ply == -1 and self.b_king_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]
            
            self.b_king_move_ply = this_move_num
            self.position_flags = self.position_flags | 2
            
            if from_sq == E8 and to_sq == G8:
                self.position_flags = self.position_flags | (1 << 4)
                self.b_kr_move_ply = this_move_num
                self.b_king_castle_ply = this_move_num

                self.occupied[Side.BLACK] ^= (0x5 << 56)
                self.pieces[PieceType.B_ROOK] ^= (0x5 << 56)
                self.squares[bit_position(H8)] = PieceType.NULL
                self.squares[bit_position(F8)] = PieceType.B_ROOK
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[56][PieceType.B_ROOK - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[58][PieceType.B_ROOK - 1]

            elif from_sq == E8 and to_sq == C8:
                self.position_flags = self.position_flags | (1 << 5)
                self.b_qr_move_ply = this_move_num
                self.b_king_castle_ply = this_move_num

                self.occupied[Side.BLACK] ^= (0x90 << 56)
                self.pieces[PieceType.B_ROOK] ^= (0x90 << 56)
                self.squares[bit_position(A8)] = PieceType.NULL
                self.squares[bit_position(D8)] = PieceType.B_ROOK
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[63][PieceType.B_ROOK - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[60][PieceType.B_ROOK - 1]

        elif piece_type == PieceType.B_ROOK:
            self.pieces[PieceType.B_ROOK] ^= (from_sq | to_sq)

            if from_sq == H8 and self.b_kr_move_ply == -1:
                self.position_flags = self.position_flags | (1 << 4)
                self.b_kr_move_ply = this_move_num
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
            elif from_sq == A8 and self.b_qr_move_ply == -1:
                self.position_flags = self.position_flags | (1 << 5)
                self.b_qr_move_ply = this_move_num
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]
                
        # self.attacks[Side.WHITE] = self.get_attacks(Side.WHITE)
        # self.attacks[Side.BLACK] = self.get_attacks(Side.BLACK)

        # update piece attacks incrementally
        # .. for the sliding pieces
        for pt in self.blocking_change(move):
            base_pt = PieceType.base_type(pt)
            pt_side = Side.WHITE if PieceType.is_white(pt) else Side.BLACK
            self.piece_attacks[pt] = self.get_piece_attacks(base_pt, pt_side)
        # .. and for the moved piece
        self.piece_attacks[piece_type] = self.get_piece_attacks(PieceType.base_type(piece_type), side)
        # .. and for any captured piece
        captured = self.squares[to_square_ind]
        if captured:
            self.piece_attacks[captured] = self.get_piece_attacks(PieceType.base_type(captured), side ^ 1)

        # recompute the attacks for each side
        for base_pt in [PieceType.P, PieceType.N, PieceType.B, PieceType.R, PieceType.Q, PieceType.K]:
            self.attacks = [0, 0]
            for pt in PieceType.piece_types(base_only=False):
                pt_side = Side.WHITE if PieceType.is_white(pt) else Side.BLACK
                self.attacks[pt_side] |= self.piece_attacks[pt]
            
        self.squares[bit_position(from_sq)] = PieceType.NULL
        self.squares[bit_position(to_sq)] = piece_type

        self.moves.append(move)

def zobrist_pieces(pieces, piece_type):
    zobrist_hash = 0
    for piece in iterate_pieces(pieces):
        square_ind = int(math.log(piece, 2))
        piece_ind = piece_type - 1
        zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[square_ind][piece_ind]
    return zobrist_hash
