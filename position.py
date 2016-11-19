from piece_type import *
from move_gen import *
from bb import *
from move import *
import tt
import math

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
            self.position_flags = Side.WHITE.value << 6

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
        self.squares = [PieceType.NULL.value for i in range(0,64)]
        self.squares[0] = self.squares[7] = PieceType.W_ROOK.value
        self.squares[1] = self.squares[6] = PieceType.W_KNIGHT.value
        self.squares[2] = self.squares[5] = PieceType.W_BISHOP.value
        self.squares[3] = PieceType.W_KING.value
        self.squares[4] = PieceType.W_QUEEN.value
        self.squares[56] = self.squares[63] = PieceType.B_ROOK.value
        self.squares[57] = self.squares[62] = PieceType.B_KNIGHT.value
        self.squares[58] = self.squares[61] = PieceType.B_BISHOP.value
        self.squares[59] = PieceType.B_KING.value
        self.squares[60] = PieceType.B_QUEEN.value

    def init_pieces(self):
        self.pieces = [None] * 13
        self.pieces[PieceType.NULL.value] = 0

        self.pieces[PieceType.W_PAWN.value] = 0xff00
        self.pieces[PieceType.W_KNIGHT.value] = 0x42
        self.pieces[PieceType.W_BISHOP.value] = 0x24
        self.pieces[PieceType.W_QUEEN.value] = 0x10
        self.pieces[PieceType.W_KING.value] = 0x8
        self.pieces[PieceType.W_ROOK.value] = 0x81
        
        self.pieces[PieceType.B_PAWN.value] = 0xff00 << 40
        self.pieces[PieceType.B_KNIGHT.value] = 0x42 << 56
        self.pieces[PieceType.B_BISHOP.value] = 0x24 << 56
        self.pieces[PieceType.B_QUEEN.value] = 0x10 << 56
        self.pieces[PieceType.B_KING.value] = 0x8 << 56
        self.pieces[PieceType.B_ROOK.value] = 0x81 << 56

    def init_occupied(self):
        self.occupied = [self.get_occupied(Side.WHITE.value),
                         self.get_occupied(Side.BLACK.value)]

    def init_attacks(self):
        self.attacks = [self.get_attacks(Side.WHITE.value),
                        self.get_attacks(Side.BLACK.value)]
        self.piece_attacks = [None] * 13
        for base_pt in [PieceType.P, PieceType.N, PieceType.B, PieceType.R, PieceType.Q, PieceType.K]:
            white = PieceType.piece(base_pt.value, Side.WHITE.value)
            black = PieceType.piece(base_pt.value, Side.BLACK.value)
            self.piece_attacks[white] = self.get_piece_attacks(base_pt.value, Side.WHITE.value)
            self.piece_attacks[black] = self.get_piece_attacks(base_pt.value, Side.BLACK.value)

    def init_zobrist(self):
        self.zobrist_hash = 0
        for piece_type, pieces in enumerate(self.pieces):
            if piece_type != PieceType.NULL.value:
                self.zobrist_hash ^= zobrist_pieces(pieces, piece_type)
        self.zobrist_hash ^= tt.ZOBRIST_SIDE[Side.WHITE.value]
        for ind, rand in enumerate(tt.ZOBRIST_CASTLE):
            self.zobrist_hash ^= rand
            
    @classmethod
    def from_fen(cls, fen):
        pieces, color, castling, en_pessant, halfmove_clock, move_num = fen.split()
        position = Position()

        # side to move
        side = Side.WHITE.value if color == "w" else Side.BLACK.value
        position_flags = side << 6
        
        # pre-init
        position.squares = [PieceType.NULL.value for i in range(0,64)]
        position.pieces = [0] * 13
        position.pieces[PieceType.NULL.value] = 0
        
        # pieces
        pieces_by_row = pieces.split("/")
        for row, pieces in enumerate(reversed(pieces_by_row)):
            col = 0
            for char in reversed(pieces):
                if char.isnumeric():
                    col += int(char) - 1
                else:
                    piece_type = None
                    color = Side.WHITE.value if char.isupper() else Side.BLACK.value
                    lchar = char.lower()
                    if lchar == "p":
                        piece_type = PieceType.piece(PieceType.P.value, color)
                    elif lchar == "n":
                        piece_type = PieceType.piece(PieceType.N.value, color)
                    elif lchar == "b":
                        piece_type = PieceType.piece(PieceType.B.value, color)
                    elif lchar == "r":
                        piece_type = PieceType.piece(PieceType.R.value, color)
                    elif lchar == "q":
                        piece_type = PieceType.piece(PieceType.Q.value, color)
                    elif lchar == "k":
                        piece_type = PieceType.piece(PieceType.K.value, color)
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
            pawn = PieceType.piece(PieceType.P.value, side ^ 1)
            if side ^ 1 == Side.WHITE.value:
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
            if PieceType.base_type(pt) not in [PieceType.B.value, PieceType.R.value, PieceType.Q.value]:
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
        side = side_to_move(self.position_flags)
        last_move = self.last_move()
        if white_to_move(self.position_flags):
            own = self.occupied[Side.WHITE.value]
            other = self.occupied[Side.BLACK.value]
            attacked = self.attacks[Side.BLACK.value]
            knights = self.pieces[PieceType.W_KNIGHT.value]
            bishops = self.pieces[PieceType.W_BISHOP.value]
            rooks = self.pieces[PieceType.W_ROOK.value]
            pawns = self.pieces[PieceType.W_PAWN.value]
            queens = self.pieces[PieceType.W_QUEEN.value]
            king = self.pieces[PieceType.W_KING.value]
            for from_sq, to_sq in pawn_moves(pawns, own, other,
                                             side, last_move.piece_type,
                                             last_move.from_sq, last_move.to_sq):
                yield Move(PieceType.W_PAWN.value, from_sq, to_sq)
            for from_sq, to_sq in knight_moves(knights, own):
                yield Move(PieceType.W_KNIGHT.value, from_sq, to_sq)
            for from_sq, to_sq in bishop_moves(bishops, own, other):
                yield Move(PieceType.W_BISHOP.value, from_sq, to_sq)
            for from_sq, to_sq in rook_moves(rooks, own, other):
                yield Move(PieceType.W_ROOK.value, from_sq, to_sq)
            for from_sq, to_sq in queen_moves(queens, own, other):
                yield Move(PieceType.W_QUEEN.value, from_sq, to_sq)
            for from_sq, to_sq in king_castle_moves(own, other, attacked, self.position_flags):
                yield Move(PieceType.W_KING.value, from_sq, to_sq)
            for from_sq, to_sq in king_moves(king, own, attacked):
                yield Move(PieceType.W_KING.value, from_sq, to_sq)
        else:
            own = self.occupied[Side.BLACK.value]
            other = self.occupied[Side.WHITE.value]
            attacked = self.attacks[Side.WHITE.value]
            knights = self.pieces[PieceType.B_KNIGHT.value]
            bishops = self.pieces[PieceType.B_BISHOP.value]
            rooks = self.pieces[PieceType.B_ROOK.value]
            pawns = self.pieces[PieceType.B_PAWN.value]
            queens = self.pieces[PieceType.B_QUEEN.value]
            king = self.pieces[PieceType.B_KING.value]
            for from_sq, to_sq in pawn_moves(pawns, own, other,
                                             side, last_move.piece_type,
                                             last_move.from_sq, last_move.to_sq):
                yield Move(PieceType.B_PAWN.value, from_sq, to_sq)
            for from_sq, to_sq in knight_moves(knights, own):
                yield Move(PieceType.B_KNIGHT.value, from_sq, to_sq)
            for from_sq, to_sq in bishop_moves(bishops, own, other):
                yield Move(PieceType.B_BISHOP.value, from_sq, to_sq)
            for from_sq, to_sq in rook_moves(rooks, own, other):
                yield Move(PieceType.B_ROOK.value, from_sq, to_sq)
            for from_sq, to_sq in queen_moves(queens, own, other):
                yield Move(PieceType.B_QUEEN.value, from_sq, to_sq)
            for from_sq, to_sq in king_castle_moves(own, other, attacked, self.position_flags):
                yield Move(PieceType.B_KING.value, from_sq, to_sq)
            for from_sq, to_sq in king_moves(king, own, attacked):
                yield Move(PieceType.B_KING.value, from_sq, to_sq)

    def last_move(self):
        return self.moves[-1] if len(self.moves) else Move(PieceType.NULL.value, None, None)

    def side_to_move(self):
        return side_to_move(self.position_flags)

    def white_to_move(self):
        return white_to_move(self.position_flags)

    def black_to_move(self):
        return black_to_move(self.position_flags)

    def in_check(self, side=None):
        if side is None:
            side = self.side_to_move()
        stm_king = PieceType.piece(PieceType.K.value, side)
        return am_in_check(self.attacks[side ^ 1], self.pieces[stm_king])
    
    def generate_moves(self):
        for move in self.get_move_candidates():
            try_move = Position(self)
            try_move.make_move(move)
            if white_to_move(try_move.position_flags):
                # black just made a move, make sure we didn't put our king in check
                if not am_in_check(try_move.attacks[Side.WHITE.value],
                                   try_move.pieces[PieceType.B_KING.value]):
                    yield move
            else:
                # white just made a move, make sure we didn't put our king in check
                if not am_in_check(try_move.attacks[Side.BLACK.value],
                                   try_move.pieces[PieceType.W_KING.value]):
                    yield move

    def get_piece_attacks(self, piece_type, side):
        if side is None:
            side = self.side_to_move()
        piece_type_side = PieceType.piece(piece_type, side)
        occupied = self.occupied[Side.WHITE.value] | self.occupied[Side.BLACK.value]
        free = occupied ^ FULL_BOARD
        if piece_type == PieceType.P.value:
            return pawn_attack(self.pieces[piece_type_side], side)
        elif piece_type == PieceType.N.value:
            return knight_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.B.value:
            return bishop_attack(self.pieces[piece_type_side], free)
        elif piece_type == PieceType.Q.value:
            return queen_attack(self.pieces[piece_type_side], free)
        elif piece_type == PieceType.K.value:
            return king_attack(self.pieces[piece_type_side])
        elif piece_type == PieceType.R.value:
            return rook_attack(self.pieces[piece_type_side], free)

    def get_attacks(self, side):
        occupied = self.occupied[Side.WHITE.value] | self.occupied[Side.BLACK.value]
        free = occupied ^ FULL_BOARD
        if side == Side.WHITE.value:
            w_pawn_attacks = pawn_attack(self.pieces[PieceType.W_PAWN.value], side)
            w_knights_attacks = knight_attack(self.pieces[PieceType.W_KNIGHT.value])
            w_bishops_attacks = bishop_attack(self.pieces[PieceType.W_BISHOP.value], free)
            w_queens_attacks = queen_attack(self.pieces[PieceType.W_QUEEN.value], free)
            w_king_attacks = king_attack(self.pieces[PieceType.W_KING.value])
            w_rooks_attacks = rook_attack(self.pieces[PieceType.W_ROOK.value], free)
            return w_pawn_attacks \
                | w_knights_attacks \
                | w_bishops_attacks \
                | w_queens_attacks \
                | w_king_attacks \
                | w_rooks_attacks
        else:
            b_pawn_attacks = pawn_attack(self.pieces[PieceType.B_PAWN.value], side)
            b_knights_attacks = knight_attack(self.pieces[PieceType.B_KNIGHT.value])
            b_bishops_attacks = bishop_attack(self.pieces[PieceType.B_BISHOP.value], free)
            b_queens_attacks = queen_attack(self.pieces[PieceType.B_QUEEN.value], free)
            b_king_attacks = king_attack(self.pieces[PieceType.B_KING.value])
            b_rooks_attacks = rook_attack(self.pieces[PieceType.B_ROOK.value], free)
            return b_pawn_attacks \
                | b_knights_attacks \
                | b_bishops_attacks \
                | b_queens_attacks \
                | b_king_attacks \
                | b_rooks_attacks

    def is_mate(self):
        if white_to_move(self.position_flags):
            if not am_in_check(self.attacks[Side.BLACK.value],
                               self.pieces[PieceType.W_KING.value]):
                return False
            for move in self.generate_moves():
                try_move = Position(self)
                try_move.make_move(move)
                if not am_in_check(try_move.attacks[Side.BLACK.value],
                                   try_move.pieces[PieceType.W_KING.value]):
                    return False
            return True
        else:
            if not am_in_check(self.attacks[Side.WHITE.value],
                               self.pieces[PieceType.B_KING.value]):
                return False
            for move in self.generate_moves():
                try_move = Position(self)
                try_move.make_move(move)
                if not am_in_check(try_move.attacks[Side.WHITE.value],
                                   try_move.pieces[PieceType.B_KING.value]):
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
        side = Side.WHITE.value if PieceType.is_white(piece_type) else Side.BLACK.value
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
            self.pieces[PieceType.B_PAWN.value] &= capture_mask
            self.pieces[PieceType.B_KNIGHT.value] &= capture_mask
            self.pieces[PieceType.B_BISHOP.value] &= capture_mask
            self.pieces[PieceType.B_ROOK.value] &= capture_mask
            self.pieces[PieceType.B_QUEEN.value] &= capture_mask
            self.occupied[Side.BLACK.value] &= capture_mask
            self.occupied[Side.WHITE.value] ^= from_sq ^ to_sq

            if to_sq & self.pieces[PieceType.B_PAWN.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_PAWN.value]
            elif to_sq & self.pieces[PieceType.B_KNIGHT.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_KNIGHT.value]
            elif to_sq & self.pieces[PieceType.B_BISHOP.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_BISHOP.value]
            elif to_sq & self.pieces[PieceType.B_ROOK.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_ROOK.value]
            elif to_sq & self.pieces[PieceType.B_QUEEN.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.B_QUEEN.value]

        elif PieceType.is_black(piece_type):
            self.pieces[PieceType.W_PAWN.value] &= capture_mask
            self.pieces[PieceType.W_KNIGHT.value] &= capture_mask
            self.pieces[PieceType.W_BISHOP.value] &= capture_mask
            self.pieces[PieceType.W_ROOK.value] &= capture_mask
            self.pieces[PieceType.W_QUEEN.value] &= capture_mask
            self.occupied[Side.WHITE.value] &= capture_mask
            self.occupied[Side.BLACK.value] ^= from_sq ^ to_sq

            if to_sq & self.pieces[PieceType.B_PAWN.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_PAWN.value]
            elif to_sq & self.pieces[PieceType.B_KNIGHT.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_KNIGHT.value]
            elif to_sq & self.pieces[PieceType.B_BISHOP.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_BISHOP.value]
            elif to_sq & self.pieces[PieceType.B_ROOK.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_ROOK.value]
            elif to_sq & self.pieces[PieceType.B_QUEEN.value] > 0:
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[to_square_ind][PieceType.W_QUEEN.value]

        if piece_type == PieceType.W_PAWN.value:
            self.pieces[PieceType.W_PAWN.value] ^= (from_sq | to_sq)

            if last_move.piece_type == PieceType.B_PAWN.value \
               and last_move.from_sq == to_sq << 8 \
               and last_move.to_sq == to_sq >> 8:
                self.occupied[Side.BLACK.value] ^= last_move.to_sq
                self.pieces[PieceType.B_PAWN.value] ^= last_move.to_sq
                self.squares[bit_position(last_move.to_sq)] = PieceType.NULL.value

        elif piece_type == PieceType.W_KNIGHT.value:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_BISHOP.value:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_QUEEN.value:
            self.pieces[piece_type] = self.pieces[piece_type] ^ (from_sq | to_sq)

        elif piece_type == PieceType.W_KING.value:
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

                self.occupied[Side.WHITE.value] ^= 0x5
                self.pieces[PieceType.W_ROOK.value] ^= 0x5
                self.squares[bit_position(H1)] = PieceType.NULL.value
                self.squares[bit_position(F1)] = PieceType.W_ROOK.value
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[0][PieceType.W_ROOK.value - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[2][PieceType.W_ROOK.value - 1]

            elif from_sq == E1 and to_sq == C1:
                self.w_qr_move_ply = this_move_num
                self.w_king_castle_ply = this_move_num
                self.position_flags = self.position_flags | (1 << 3)

                self.occupied[Side.WHITE.value] ^= 0x90
                self.pieces[PieceType.W_ROOK.value] ^= 0x90
                self.squares[bit_position(A1)] = PieceType.NULL.value
                self.squares[bit_position(D1)] = PieceType.W_ROOK.value
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[7][PieceType.W_ROOK.value - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[4][PieceType.W_ROOK.value - 1]

        elif piece_type == PieceType.W_ROOK.value:
            self.pieces[PieceType.W_ROOK.value] ^= (from_sq | to_sq)

            if from_sq == H1 and self.w_kr_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[0]
                self.position_flags = self.position_flags | (1 << 2)
                self.w_kr_move_ply = this_move_num
            elif from_sq == A1 and self.w_qr_move_ply == -1:
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[1]
                self.position_flags = self.position_flags | (1 << 3)
                self.w_qr_move_ply = this_move_num

        elif piece_type == PieceType.B_PAWN.value:
            self.pieces[PieceType.B_PAWN.value] ^= (from_sq | to_sq)

            if last_move.piece_type == PieceType.W_PAWN.value \
               and last_move.from_sq == to_sq >> 8 \
               and last_move.to_sq == to_sq << 8:
                self.occupied[Side.WHITE.value] ^= last_move.to_sq
                self.pieces[PieceType.W_PAWN.value] ^= last_move.to_sq
                self.squares[bit_position(last_move.to_sq)] = PieceType.NULL.value

        elif piece_type == PieceType.B_KNIGHT.value:
            self.pieces[PieceType.B_KNIGHT.value] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_BISHOP.value:
            self.pieces[PieceType.B_BISHOP.value] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_QUEEN.value:
            self.pieces[PieceType.B_QUEEN.value] ^= (from_sq | to_sq)

        elif piece_type == PieceType.B_KING.value:
            self.pieces[PieceType.B_KING.value] ^= (from_sq | to_sq)

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

                self.occupied[Side.BLACK.value] ^= (0x5 << 56)
                self.pieces[PieceType.B_ROOK.value] ^= (0x5 << 56)
                self.squares[bit_position(H8)] = PieceType.NULL.value
                self.squares[bit_position(F8)] = PieceType.B_ROOK.value
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[56][PieceType.B_ROOK.value - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[58][PieceType.B_ROOK.value - 1]

            elif from_sq == E8 and to_sq == C8:
                self.position_flags = self.position_flags | (1 << 5)
                self.b_qr_move_ply = this_move_num
                self.b_king_castle_ply = this_move_num

                self.occupied[Side.BLACK.value] ^= (0x90 << 56)
                self.pieces[PieceType.B_ROOK.value] ^= (0x90 << 56)
                self.squares[bit_position(A8)] = PieceType.NULL.value
                self.squares[bit_position(D8)] = PieceType.B_ROOK.value
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[63][PieceType.B_ROOK.value - 1]
                self.zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[60][PieceType.B_ROOK.value - 1]

        elif piece_type == PieceType.B_ROOK.value:
            self.pieces[PieceType.B_ROOK.value] ^= (from_sq | to_sq)

            if from_sq == H8 and self.b_kr_move_ply == -1:
                self.position_flags = self.position_flags | (1 << 4)
                self.b_kr_move_ply = this_move_num
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[2]
            elif from_sq == A8 and self.b_qr_move_ply == -1:
                self.position_flags = self.position_flags | (1 << 5)
                self.b_qr_move_ply = this_move_num
                self.zobrist_hash ^= tt.ZOBRIST_CASTLE[3]
                
        # self.attacks[Side.WHITE.value] = self.get_attacks(Side.WHITE.value)
        # self.attacks[Side.BLACK.value] = self.get_attacks(Side.BLACK.value)

        # update piece attacks incrementally
        # .. for the sliding pieces
        for pt in self.blocking_change(move):
            base_pt = PieceType.base_type(pt)
            pt_side = Side.WHITE.value if PieceType.is_white(pt) else Side.BLACK.value
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
                pt_side = Side.WHITE.value if PieceType.is_white(pt) else Side.BLACK.value
                self.attacks[pt_side] |= self.piece_attacks[pt]
            
        self.squares[bit_position(from_sq)] = PieceType.NULL.value
        self.squares[bit_position(to_sq)] = piece_type

        self.moves.append(Move(piece_type, from_sq, to_sq))

def zobrist_pieces(pieces, piece_type):
    zobrist_hash = 0
    for piece in iterate_pieces(pieces):
        square_ind = int(math.log(piece, 2))
        piece_ind = piece_type - 1
        zobrist_hash ^= tt.ZOBRIST_PIECE_SQUARES[square_ind][piece_ind]
    return zobrist_hash
