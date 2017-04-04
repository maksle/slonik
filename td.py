from IPython import embed
import random
import time
from search import allowance_to_depth
from position import Position
from side import Side
from search import Engine
from collections import namedtuple
from features import ToFeature
from evals import BaseEvaluator
from nn import nn_fit, nn_predict
import numpy as np
import logging
import logging_config


log = logging.getLogger(__name__)

toFeature = ToFeature()

def get_features(position):
    toFeature.set_position(position)
    return toFeature.ann_features()

def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

def initialize_weights(positions):
    features = [get_features(psn) for psn in positions]
    scores = [BaseEvaluator(psn).evaluate() for psn in positions]
    scores = [min(1, max(-1, score / 1000)) for score in scores]
    nn_fit(features, scores, 10, batch_size)

def nn_evaluator(position):
    features = get_features(position)
    return nn_predict(features)

    
# root_positions = total_fens = 700000
# plies_to_play = 12
# batch = 256
# num_targets_per_iteration = batch * plies_to_play = 3072        
# num_iterations = total_fens / batch = 2734

depth = 4.5
td_lambda = 0.7
position_value = namedtuple('position_value', ['fen', 'leaf_val', 'features'])

iterations = 10000
total_fens = 700762
plies_to_play = 16
batch_size = 2 #32
positions_per_iteration = 12 #256
num_iterations = total_fens // positions_per_iteration + 1

if __name__ == "__main__":
    offset = 0
    for itern in range(num_iterations):
        positions = []
        lines_read = 0
        with open('../allfens.txt') as fens:
            while lines_read != offset:
                fen = fens.readline()
                lines_read += 1
                if fen == '': break
            lines_read = 0
            while lines_read != positions_per_iteration:
                fen = fens.readline()
                lines_read += 1
                if fen == '': break
                positions.append(Position.from_fen(fen))
            offset += positions_per_iteration

        if itern == 0:
            initialize_weights(positions)
        else:
            for psn in positions:
                print(psn.fen())
                moves = list(psn.generate_moves_all(legal=True))
                move = random.choice(moves)
                psn.make_move(move)

                engine = Engine()
                engine.init_move_history()
                engine.max_depth = depth
                engine.root_position = psn
                engine.info = lambda *args: True
                engine.debug_info = lambda *args: True
                engine.evaluate = nn_evaluator
                engine.search_stats.time_start = time.time()

                timesteps = []
                for ply in range(plies_to_play):
                    leaf_val, si = engine.iterative_deepening()
                    if psn.side_to_move() == Side.B:
                        leaf_val = -leaf_val
                    fen = psn.fen()
                    timesteps.append(position_value(fen, leaf_val, None))
                    if not has_moves(psn):
                        break
                    pv = si[0].pv
                    psn.make_move(pv[0])

                print("leafvals", [t.leaf_val for t in timesteps])

                targets = []
                features = []
                T = len(timesteps) - 1
                for t, data in enumerate(timesteps):
                    fen, target, _ = data
                    target = leaf_val
                    L = td_lambda
                    j = t
                    while j < T:
                        target += (timesteps[j+1].leaf_val - timesteps[j].leaf_val) * L
                        j += 1
                        L *= td_lambda
                    # targets.append(position_value(fen, target, ToFeature(psn).ann_features()))
                    features.append(get_features(psn))
                    targets.append(target)

                print("targets", targets)

            error = nn_fit(features, targets, 10, batch_size)

            # epoch_iterations = len(targets) // batch_size + 1
            # for i in range(epochs):
            #     total_error = 0
            #     for j in range(epoch_iterations):
            #         ann_features = []
            #         ann_targets = []
            #         sample = random.sample(targets, batch_size)
            #         for k in range(batch_size):
            #             ann_features.append(sample.features)
            #             ann_targets.append(sample.target)
            #         history = ann_fit(ann_features, ann_targets)
            #         total_error += history
            #     print("Epoch error:", total_error / epoch_iterations)

        if itern % 20 == 0:
            pass # do a test
