from IPython import embed
import random
import time
from search import allowance_to_depth
from position import Position
from side import Side
from search import Engine
from collections import namedtuple
from evals import BaseEvaluator
from nn import model
import nn_evaluate
import numpy as np
import logging
import logging_config


log = logging.getLogger(__name__)


def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

def initialize_weights(positions):
    features = [nn_evaluate.get_features(psn) for psn in positions]
    scores = []
    for psn in positions:
        s = BaseEvaluator(psn).evaluate()
        if psn.black_to_move():
            s = -s
        scores.append(s)
    scores = [min(1, max(-1, score / 1000)) for score in scores]
    model.fit(features, scores, 10, batch_size)

# root_positions = total_fens = 700000
# plies_to_play = 12
# batch = 256
# num_targets_per_iteration = batch * plies_to_play = 3072        
# num_iterations = total_fens / batch = 2734

depth = 5 #6.5
# td_lambda = 0.7
# position_value = namedtuple('position_value', ['fen', 'leaf_val', 'features'])

iterations = 10000
total_fens = 700762
plies_to_play = 16
batch_size = 32
positions_per_iteration = 15 #128 #256
num_iterations = total_fens // positions_per_iteration + 1

if __name__ == "__main__":
    offset = 0
    for itern in range(num_iterations):
        positions = []
        lines_read = 0
        if itern == 0:
            npos = 100
            # npos = positions_per_iteration
        else:
            npos = positions_per_iteration
        with open('../allfens.txt') as fens:
            while lines_read != offset:
                fen = fens.readline()
                lines_read += 1
                if fen == '': break
            lines_read = 0
            while lines_read != npos:
                fen = fens.readline()
                lines_read += 1
                if fen == '': break
                positions.append(Position.from_fen(fen))
            offset += npos

        if itern == 0:
            # pass
            initialize_weights(positions)
            model.save_model()
            break
        else:
            for psn in positions:
                print("================")
                print(psn.fen())
                moves = list(psn.generate_moves_all(legal=True))
                move = random.choice(moves)
                print(move)
                psn.make_move(move)
                print(psn)
                
                engine = Engine()
                engine.init_move_history()
                # limits, whichever comes first
                engine.max_depth = depth
                engine.movetime = 120
                engine.root_position = psn
                engine.info = lambda *args: True
                # engine.info = print
                engine.debug_info = lambda *args: True
                # engine.debug_info = print
                engine.evaluate = nn_evaluate.evaluate
                engine.search_stats.node_count = 0
                engine.search_stats.time_start = time.time()

                model.reset_eligibility_trace()
                
                timesteps = []
                for ply in range(plies_to_play):
                    if not has_moves(psn):
                        break
                    engine.stop_event.clear()
                    engine.rotate_killers()
                    engine.search_stats.time_start = time.time()
                    leaf_val, si = engine.iterative_deepening()
                    if psn.side_to_move() == Side.B:
                        leaf_val = -leaf_val
                    leaf_val /= 1000
                    leaf_val = min(max(leaf_val, -2), 2) # mate score

                    timesteps.append(leaf_val)
                    if len(timesteps) > 2:
                        model.train(nn_evaluate.get_features(psn), timesteps[-3], timesteps[-1])
                    
                    pv = si[0].pv
                    print(pv[0])
                    psn.make_move(pv[0])
                model.save_model()
                # error = model.train(features, errors)
        
        # if itern % 20 == 0:
        #     pass # do a test and or save model
