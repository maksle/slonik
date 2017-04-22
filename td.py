from IPython import embed
import random
import time
import os
from tt import TTEntry
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
import sts


log = logging.getLogger(__name__)


def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

def fifty_move_draw(pos):
    return pos.halfmove_clock >= 50

def three_fold_repetition(pos):
    return pos.three_fold[pos.fen(timeless=True)] >= 3

def game_over(pos):
    return not has_moves(pos) or fifty_move_draw(pos) or three_fold_repetition(pos)
    
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

depth = 5.5 #6.5
# td_lambda = 0.7
# position_value = namedtuple('position_value', ['fen', 'leaf_val', 'features'])

iterations = 10000
total_fens = 700762
plies_to_play = 16
batch_size = 32
positions_per_iteration = 8 #128 #256
num_iterations = total_fens // positions_per_iteration + 1

if __name__ == "__main__":
    offset = 650
    init_npos = 20000
    sts_scores = []
    for itern in range(num_iterations):
        positions = []
        lines_read = 0
        initialize_network = itern == 0 and not os.path.exists('checkpoint')
        if initialize_network:
            npos = init_npos
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

        if initialize_network:
            # pass
            initialize_weights(positions)
            model.save_model()
            break
        else:
            n = offset
            m = 0
            for psn in positions:
                n += 1
                m += 1
                print("============================")
                # is_dyna = m > positions_per_iteration
                is_dyna = False
                if is_dyna:
                    print("New Game, dyna position")
                else:
                    print("New Game, #{0}, (#{1}/{2})".format(n, m, positions_per_iteration))
                print(psn.fen())
                
                if not is_dyna:
                    moves = list(psn.generate_moves_all(legal=True))
                    move = random.choice(moves)
                    print("Random move:", move)
                    psn.make_move(move)

                print(psn)
                    
                engine = Engine()
                engine.init_move_history()
                TTEntry.next_game()
                model.reset_eligibility_trace()
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
                
                timesteps = []
                dyna_threshold = .2
                dyna_counter = 0

                while True:
                # for ply in range(plies_to_play):
                    if game_over(psn):
                        break
                    # dyna position
                    # if is_dyna and len(psn.moves) == 16: 
                    #     break
                    engine.stop_event.clear()
                    engine.rotate_killers()
                    engine.search_stats.time_start = time.time()
                    leaf_val, si = engine.iterative_deepening()
                    if psn.side_to_move() == Side.B:
                        leaf_val = -leaf_val
                    leaf_val /= 1000
                    leaf_val = min(max(leaf_val, -1), 1) # mate score

                    pv = si[0].pv
                    leaf = Position(psn)
                    for move in pv:
                        leaf.make_move(move)
                    print(pv[0], leaf_val)
                    timesteps.append([leaf_val, leaf])

                    psn.make_move(pv[0])
                    if len(timesteps) % 10 == 0:
                        print(psn)
                    
                    # if len(timesteps) > 2:
                    #     t0_val, t0_fen = timesteps[-3]
                    #     t0_pos = Position.from_fen(t0_fen)
                    #     t1_val, t1_fen = timesteps[-1]

                    #     # This happens b/c of previous update
                    #     # val = nn_evaluate.evaluate(t0_pos)
                    #     # if t0_pos.side_to_move() == Side.B:
                    #     #     val = -val
                    #     # if abs(val/1000 - t0_val) > .0008:
                    #     #     embed()

                    #     print('             ', format(t1_val, '.4f'))
                    #     # model.train(nn_evaluate.get_features(t0_pos), t1_val)
                    #     # if abs(t1_val - t0_val) > dyna_threshold and dyna_counter < 1 and len(positions) < 16:
                    #     #     fen = timesteps[-3][1]
                    #     #     dyna_pos = Position.from_fen(fen)
                    #     #     positions.append(dyna_pos)
                    #     #     dyna_counter += 1
                    #     #     print("Adding dyna position:\n")
                    #     #     print(dyna_pos)
                    
                    # print()
                    
                positions = []
                targets = []
                        
                lamda = .7
                T = len(timesteps)
                for ind, t in enumerate(timesteps):
                    leaf_val, pos = t
                    error = 0
                    L = 1
                    for j in range(ind+2, T, 2):
                        L *= lamda
                        delta = timesteps[j][0] - timesteps[j-2][0]
                        error += L * delta
                    target = error + leaf_val
                    positions.append(pos)
                    targets.append(target)
            
            
                features = [nn_evaluate.get_features(psn) for psn in positions]
                model.fit(features, targets, 10, batch_size)
                model.save_model()
                
        # if itern % 20 == 0:
        if True:
            sts_score = sts.run_sts_test()
            sts_scores.append(sts_score)
            model.update_sts_score(sts_score)
            print("STS scores over time:", sts_scores)

