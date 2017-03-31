import random

iterations = 1000000
plies_to_play = 64
depth = allowance_to_depth(512)

for itern in iterations:
    if itern == 0:
        for psn in positions:
            score = Evaluation(psn).init_attacks().evaluate()
            nn.train(positions, score)
    else:
        for psn in positions:
            moves = psn.generate_moves_all(legal=True)
            move = random.choice(moves)
            psn.make_move(move)

            engine.init_move_history()
            engine.depth = depth
            engine.search_stats.time_start = time.time()

            for ply in range(plies_to_play):
                engine.root_position = psn
                val, si = engine.iterative_deepening()
