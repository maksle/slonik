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
from material import material_bootstrap
from evals import BaseEvaluator, arbiter_draw, fifty_move_draw, three_fold_repetition
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

def game_over(pos):
    return not has_moves(pos) or arbiter_draw(pos)
    
def initialize_weights(positions):
    features = [nn_evaluate.get_features(psn) for psn in positions]
    scores = []
    for psn in positions:
        # s = BaseEvaluator(psn).evaluate()
        s = material_bootstrap(psn)
        # if psn.black_to_move():
        #     s = -s
        s = min(1, max(-1, s / 1000))
        scores.append(s)
    model.fit(features, scores, 10, batch_size)

depth = 2.5 #6.5
iterations = 10000
total_fens = 700762
plies_to_play = 32
positions_per_iteration = 128 #128 #256
batch_size = plies_to_play * positions_per_iteration // 8
num_iterations = total_fens // positions_per_iteration + 1
epochs = 10

if __name__ == "__main__":
    offset = 20000
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

            update_features = []
            update_targets = []
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

                # while True:
                for ply in range(plies_to_play):
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
                    leaf_val = min(max(leaf_val, -1), 1)

                    pv = si[0].pv
                    leaf = Position(psn)
                    for move in pv:
                        leaf.make_move(move)
                    print(pv[0], leaf_val)

                    is_nn_eval = True
                    # The leaf eval should be the same or it's a game end result.
                    # We don't want to reward/penalize the nn for evals it didn't make,
                    # except for end of game rewards that are deducable from the position
                    # (but not draws due to 50 move rule for example). If we include move
                    # count in the nn features, would have to also include it in the zhash.
                    # jval, jpos = timesteps[j]
                    check_val = nn_evaluate.evaluate(leaf)
                    if leaf.side_to_move() == Side.B:
                        check_val *= -1
                    if abs(check_val/1000 - leaf_val) > .0008:
                        is_nn_eval = False
                        # if fifty_move_draw(pos) or three_fold_repetition(pos):
                        #     continue
                    
                    timesteps.append([leaf_val, leaf, is_nn_eval])
                    if not is_nn_eval:
                        break

                    psn.make_move(pv[0])
                    if len(timesteps) % 10 == 0:
                        print(psn)
                
                lamda = .7
                T = len(timesteps)
                for ind, t in enumerate(timesteps):
                    leaf_val, pos, is_nn_eval = t
                    
                    if not is_nn_eval:
                        target = leaf_val
                    else:
                        error = 0
                        L = 1
                        for j in range(ind+2, T, 2):
                            L *= lamda
                            delta = timesteps[j][0] - timesteps[j-2][0]
                            error += L * delta
                        target = error + leaf_val
                    update_features.append(nn_evaluate.get_features(pos))
                    update_targets.append(target)
            
            model.fit(update_features, update_targets, epochs, batch_size)
            model.save_model()
                
        # if itern % 20 == 0:
        if True:
            sts_score = sts.run_sts_test()
            sts_scores.append(sts_score)
            model.update_sts_score(sts_score)
            print("STS scores over time:", sts_scores)

