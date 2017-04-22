import chess
import chess.pgn
import random


pgn = open('/home/maks/scid/allgames.pgn')
while True:
    game = chess.pgn.read_game(pgn)
    if game is None: break
    if len(game.errors):
        continue
    moves = list(game.main_line())
    if len(moves) < 10:
        continue
    stop = random.randrange(0, len(moves))
    b = game.board()
    for move in moves[:stop]:
        b.push(move)
    print(b.fen())
