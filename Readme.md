# Slonik - Chess engine
Slonik is a chess engine written in Python. It is command-line driven, and will soon plug into chess GUIs, by implementing the UCI interface.

## Why in Python?
I chose Python to iterate quickly. It has allowed me to learn and implement more quickly, though Python is considerably slow for this domain.

## Features / Ideas
The ideas for this engine are largely the ideas of existing engines out there. Chessprogramming.wikispaces.com has been an especially useful resource, in addition to various academic articles in the field of chess AI. The source code for the well known open-source engine, Stockfish, and the recent ground-breaking chess engine, Giraffe (and the white-paper describing it's ideas), were especially useful. I took from Giraffe the idea for a probabilistic based search, though many ideas in Slonik are implemented from a depth based mindset. The engine features the following:

- minimax search (negamax) with alpha-beta pruning
- quiescence search
- iterative deepening and internal iterative deepening
- probablistic based search
- aspiration windows and principal variation search
- null move search and reduced depth/probability search
- late move reduction implicit via probability search
- single-move extension
- check, capture evasion, recapture, and other extensions
- futility reductions
- killer move heuristic (ply-based good moves)
- counter move heuristic
- history heuristic (piece/square keys of good moves)
- piece-square table values

# Plans
The major plan for this engine is to try out more novel ideas, probably on top of the current ideas, in a simliar vein to what the chess engine Giraffe has done. That is probably the use of Temporal Difference Learning, and other reinforcement learning ideas. I may also look into porting this to a faster language, or perhaps implement parts of it in Cython.

- complete the UCI implementation
- evaluation tuning, and AI evaluation
- speed optimizations and/or port to another language
