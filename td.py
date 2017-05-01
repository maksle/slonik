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
import math
import nn_evaluate
import numpy as np
import logging
import logging_config
import sts


log = logging.getLogger(__name__)


class Timestep(object):
    current_target_generation = 0
    def __init__(self, leaf, features, target_val, target_gen):
        self.leaf = leaf
        self.features = features
        self.target_val = target_val
        self.target_gen = target_gen

    def update_target_val(self):
        if self.target_val is None or self.target_gen < Timestep.current_target_generation:
            self.target_val = model.target.predict(self.features)
            self.target_gen = Timestep.current_target_generation

        
def batch_indexes(size, batch_size):
    num_batches = int(math.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

def game_over(pos):
    return not has_moves(pos) or arbiter_draw(pos)

def eval_is_fixed(leaf, eval_from_search):
    """The leaf eval should theoretically be the same as the result of the
    search, or it's a game end result (fixed)."""
    check_val = nn_evaluate.evaluate(leaf)
    if leaf.side_to_move() == Side.B:
        check_val *= -1
    check_val = min(max(check_val/1000, -1), 1)
    if abs(check_val - eval_from_search) > .0008:
        return True
    return False
   
def sum_TD_errors(timesteps):
    lamda = .7
    features = []
    targets = []

    T = len(timesteps)
    for ind, t in enumerate(timesteps):
        t.update_target_val()

        error = 0
        L = 1
        for j in range(ind+2, T, 2):
            L *= lamda

            # target value comes from the target network for an unbaised evaluation
            t_next = timesteps[j]
            t_next.update_target_val()

            t_prev = timesteps[j-2]
            t_prev.update_target_val()

            delta = t_next.target_val - t_prev.target_val
            error += L * delta
        
        target = error + t.target_val
        
        targets.append(target)
        features.append(t.features)

    return features, targets

def initialize_weights(positions):
    features = [nn_evaluate.get_features(psn) for psn in positions]
    scores = []
    for psn in positions:
        # s = BaseEvaluator(psn).evaluate()
        s = material_bootstrap(psn)
        s = min(1, max(-1, s / 1000))
        scores.append(s)
    for _ in range(10):
        z = list(zip(features, scores))
        random.shuffle(z)
        rfeatures, rscores = list(zip(*z))
        for start, stop in batch_indexes(len(positions), 32):
            model.actor.train_batch(rfeatures[start:stop], rscores[start:stop])
    model.copy_to_target()
    Timestep.current_target_generation += 1

def train(episodes, write_summary):
    features = []
    targets = []
    for timesteps in episodes:
        f, t = sum_TD_errors(timesteps)
        features.extend(f)
        targets.extend(t)

    # get random batch
    z = list(zip(features, targets))
    random.shuffle(z)
    rfeatures, rtargets = list(zip(*z))
    bfeatures, btargets = rfeatures[:batch_size], rtargets[:batch_size]

    # update the actor
    model.actor.train_batch(bfeatures, btargets, write_summary)


if __name__ == "__main__":
    depth = 3 #6.5
    total_fens = 700762
    plies_to_play = 32
    positions_per_iteration = 256
    batch_size = 32 # plies_to_play * positions_per_iteration // 8
    num_iterations = total_fens // positions_per_iteration + 1
    max_replay_buffer_size = 64

    offset = 50300
    init_npos = 10000
    sts_scores = []
    episodes = []
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
                print("New Game, #{0}, (#{1}/{2})".format(n, m, positions_per_iteration))
                print(psn.fen())
                
                moves = list(psn.generate_moves_all(legal=True))
                move = random.choice(moves)
                print("Random move:", move)
                psn.make_move(move)

                print(psn)
                    
                engine = Engine()
                engine.init_move_history()
                TTEntry.next_game()
                engine.max_depth = depth
                engine.movetime = 120
                engine.root_position = psn
                engine.info = lambda *args: True
                engine.debug_info = lambda *args: True
                engine.evaluate = nn_evaluate.evaluate
                engine.search_stats.node_count = 0
                engine.search_stats.time_start = time.time()
                
                timesteps = []

                for ply in range(plies_to_play):
                    if game_over(psn):
                        break

                    # do the search
                    engine.stop_event.clear()
                    engine.rotate_killers()
                    engine.search_stats.time_start = time.time()
                    leaf_val, si = engine.iterative_deepening()
                    if psn.side_to_move() == Side.B:
                        leaf_val = -leaf_val
                    leaf_val /= 1000
                    leaf_val = min(max(leaf_val, -1), 1)

                    # eval will be on the leaf
                    pv = si[0].pv
                    leaf = Position(psn)
                    for move in pv:
                        leaf.make_move(move)
                    print(pv[0], leaf_val)

                    timesteps.append(Timestep(leaf, nn_evaluate.get_features(leaf), None, -1))
                    
                    if len(episodes) == max_replay_buffer_size:
                        write_summary = ply == 0 and m == 1
                        train(episodes, write_summary=write_summary)
                    
                    # stop playing when the results are no longer the raw NN output
                    is_fixed = eval_is_fixed(leaf, leaf_val)
                    if is_fixed:
                        break

                    psn.make_move(pv[0])
                    if len(timesteps) % 10 == 0:
                        print(psn)
                
                episodes.append(timesteps)
                
                # replay buffer size is limited
                if len(episodes) > max_replay_buffer_size:
                    episodes.pop(0)
                
            # After each playing iteration:
            # .. check our progress
            sts_score = sts.run_sts_test()
            sts_scores.append(sts_score)
            model.update_sts_score(sts_score)
            print("STS scores over time:", sts_scores)
            # .. update the target network and save the models
            model.copy_to_target()
            Timestep.current_target_generation += 1
            model.save_model()
