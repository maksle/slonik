import chess
import chess.pgn
import chess.uci
import random

DEPTH = 3
STOCKFISH = "/home/maks/prog/Chess_Engines/Stockfish/stockfish-6-linux/Linux/x86-32/stockfish"

engine = chess.uci.popen_engine(STOCKFISH)
info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

with open('/home/maks/projects/stockfish_init_fens3.txt') as fens:
    for fen in fens:
        fen = fen.strip()
        
        engine.ucinewgame()
        board = chess.Board(fen=fen)
        move = random.choice(list(board.legal_moves))
        board.push(move)
        # fen = board.fen()
        engine.position(board)
        engine.go(depth=DEPTH)
        
        score = info_handler.info["score"][1].cp
        if score is None:
            if info_handler.info["score"][1].mate != 0:
                continue
            if info_handler.info["score"][1].mate > 0:
                score = 10000
            else:
                score = -10000

        wtm = board.turn
        if not wtm: score *= -1
                
        pv = info_handler.info["pv"]
        if not pv.get(1): continue
        for move in pv[1]:
            board.push(move)
        fen = board.fen()
        
        print(fen)
        print(score)
