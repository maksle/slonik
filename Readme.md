# Slonik - Chess engine
Slonik is a chess engine written in Python. It is command-line driven, and will soon be able to plug in to chess GUIs, by implementing the UCI interface, which is almost done.

## Why in Python?
I chose Python because I wanted to try be able to iterate quickly. I now somewhat regret it, because speed is important for a chess program, and Python is considerably slow for this domain. It has allowed me to learn a lot quickly though.

## Features / Ideas
The ideas for this engine are largely the ideas of existing engines out there. Chessprogramming.wikispaces.com is especially useful and the engine would not be as good as it is currently without it. I have also gotten a lot of help by looking at the source code for the well known open-source engine, Stockfish. Research articles on chess have also been read in the development of this engine. Of note was also the chess engine and white paper for Giraffe, which has a very novel and successful approach. I took from it the idea for a probabilistic based search, rather than depth based. Many ideas though are implemented from a depth based mindset. The engine features the following:

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
