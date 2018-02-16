# Slonik - Chess engine

This project is currently mainly a prototyping and testing ground for changes that are eventually to end up in my slonikai c++ project. Current work is mainly in the MCTS routine and the neural network architecture experiments in the notebooks folder. I'm currently working on bring in ideas from AlphaGo paper and from the "Thinking Fast and Slow with Deep Learning and Tree Search" ExIt RL paper.

-- 

Slonik is a UCI chess engine, which means that it plays chess. To play against it or analyze positions with it, plug it into a graphical chess interface, such as Scid vs PC or Chessbase. Slonik means "little elephant" in Russian.

There are two versions of the engine. The search can use the original static evaluation function (linear combination of hand-coded features), or it can use the new evaluator which uses a neural network and was trained with supervised learning and temporal difference learning. The SL/TD version slightly surpasses the original version now.

## Why in Python?
I chose Python to iterate quickly, which has allowed me to learn and implement more ideas, though Python is considerably slow for this domain.
One of the files is now Cythonized for about 4x speedup but the engine is still considerably slow compared to engines written in C/C++.

## Features / Ideas
The ideas for this engine are largely the ideas of existing engines out there. Chessprogramming.wikispaces.com has been an especially useful resource, in addition to various academic articles in the field of chess AI. The source code for the well known open-source engine, Stockfish, and the recent ground-breaking chess engine, Giraffe (and the white-paper describing it's ideas), were especially useful. I took from Giraffe the idea for a probabilistic based search, though many ideas in Slonik are implemented from a depth based mindset. Additionally, Giraffe's implementation of the TD learning was very helpful.

From the traditional set of features that are used in engines, this Slonik features the following:

- minimax search (negamax) with alpha-beta pruning
- quiescence search
- iterative deepening and internal iterative deepening
- probablistic based search
- transposition hash table
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

Slonik uses magic bitboards, a perfect hashing algorithm using constructive collisions of piece-square and board occupancy combinations. Better magic numbers that encode more collisions with less bits are known, but I find my own magic numbers (see `find_magic.py`). Magic bitboards were a small improvement over the kogge-stone algorithm for sliding piece move generation. Another major change was piece attacks. Previously Slonik used incremental piece attacks, and it now calculates piece attacks at position evaluation time. Unclear to me if that was a win or not.

For the AI, Slonik uses a neural network similar in structure to the one described in the white paper for Giraffe. The net is written with Tensorflow. Initialization is done on stockfish labels and then it learns from self-play. I am experimenting with variations on the TD learning. For example, there is the batch update version, and I have experimented on a fully incremental online implementation with eligibility traces, similar to what was used in TD-Gammon. I am also experimenting with using two networks for double learning, to reduce bias. There are many ideas I haven't tried yet, and some I need to come back to.

STS test suite of chess positions, which tests positional understanding, is used to evaluate progress.
