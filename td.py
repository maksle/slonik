import random
import time
from search import allowance_to_depth
from position import Position
from side import Side
from search import Engine
from collections import namedtuple
from features import ToFeature
from evals import Evaluation
import logging
import logging_config


log = logging.getLogger(__name__)

def ann_train(a,b):
    return 0
    
def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

# root_positions = total_fens = 700000
# plies_to_play = 12
# batch = 256
# num_targets_per_iteration = batch * plies_to_play = 3072        
# num_iterations = total_fens / batch = 2734

iterations = 10000
batch_size = 32

depth = 4.5
td_lambda = 0.7
position_value = namedtuple('position_value', ['fen', 'leaf_val', 'features'])

total_fens = 700762
plies_to_play = 16
positions_per_iteration = 256
num_iterations = total_fens // positions_per_iteration + 1

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
        features = [ToFeature(psn).ann_features() for psn in positions]
        scores = [Evaluation(psn).init_attacks().evaluate() for psn in positions]
        data = list(zip(features, scores))
        nbatches = len(features) // batch_size + 1
        for i in range(3):
            total_error = 0
            for j in range(nbatches):
                sample = random.sample(data, batch_size)
                f, s = zip(*sample)
                error = ann_train(f, s)
                total_error += error
            print("Epoch:", i, "error:", total_error / nbatches)
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
            engine.search_stats.time_start = time.time()
            
            timesteps = []
            for ply in range(plies_to_play):
                leaf_val, si = engine.iterative_deepening()
                if psn.side_to_move() == Side.B:
                    leaf_val = -leaf_val # need to also normalize it 
                fen = psn.fen()
                timesteps.append(position_value(fen, leaf_val, None))
                if not has_moves(psn):
                    break
                pv = si[0].pv
                psn.make_move(pv[0])

            targets = []
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
                error = ann_train(ann_features, ann_targets)
                total_error += error
            print("Epoch error:", total_error / epoch_iterations)

    if itern % 20 == 0:
        pass # do a test
