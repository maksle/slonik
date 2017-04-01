import random

iterations = 10000
batch_size = 32
plies_to_play = 64
depth = allowance_to_depth(512)
td_lambda = 0.7
position_value = namedtuple('position_value', ['fen', 'leaf_val', 'features'])

for itern in iterations:
    if itern == 0:
        features = [ToFeature(psn).ann_features() for psn in positions]
        scores = [Evaluation(psn).init_attacks().evaluate() for psn in positions]
        data = list(zip(features, scores))
        nbatches = len(features) / batch_size + 1
        for i in range(3):
            total_error = 0
            for j in range(nbatches):
                sample = random.choice(data, batch_size)
                f, s = zip(*sample)
                error = nn.train(f, s)
                total_error += error
            print("Epoch:", epoch, "error:", total_error / nbatches)
    else:
        for psn in positions:
            moves = psn.generate_moves_all(legal=True)
            move = random.choice(moves)
            psn.make_move(move)

            engine.init_move_history()
            engine.depth = depth
            engine.root_position = psn
            engine.search_stats.time_start = time.time()
            
            timesteps = []
            for ply in range(plies_to_play):
                leaf_val, si = engine.iterative_deepening()
                if psn.side_to_move() == Side.B:
                    leaf_val = -leaf_val # need to also normalize it 
                fen = psn.fen()
                timesteps.append(position_value(fen, leaf_val, None))
                if psn.game_status() != ACTIVE:
                    break
                pv = si[0].pv
                psn.make_move(pv[0])

            targets = []
            T = len(timesteps)
            for t, data in enumerate(timesteps):
                fen, target = data
                target = leaf_val
                L = td_lambda
                j = t
                while j < T:
                    target += (timesteps[j+1] - timesteps[j]) * L
                    j += 1
                    L *= td_lambda
                targets.append(position_value(fen, target, ToFeature(psn).ann_features()))
            
            epochs = 10
            epoch_iterations = len(targets) // batch_size + 1
            for i in range(epochs):
                total_error = 0
                for j in range(epoch_iterations):
                    ann_features = []
                    ann_targets = []
                    sample = random.sample(targets, batch_size)
                    for k in range(batch_size):
                        ann_features.append(sample.features)
                        ann_targets.append(sample.target)
                    error = ann.train(ann_features, ann_targets)
                    total_error += error
                print("Epoch error:", total_error / epoch_iterations)

            if iteration % 20 == 0:
                pass # do a test
            
