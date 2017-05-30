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
    def __init__(self, leaf, features, static_val, static_gen, abs_error, target, adjusted_target):
        self.leaf = leaf
        self.features = features
        self.static_val = static_val
        self.static_gen = static_gen
        self.abs_error = abs_error
        self.target = target
        self.adjusted_target = adjusted_target

    def update_static_val(self):
        if self.static_val is None or self.static_gen < Timestep.current_target_generation:
            self.static_val = model.target.predict(self.features)
            self.static_gen = Timestep.current_target_generation

        
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
        if abs(eval_from_search) != 0 and abs(eval_from_search) != 1:
            print("fixed.. searchval:", eval_from_search, "nnval:", check_val, "fen", leaf.fen())
        return True
    return False

def sample(timesteps):
    """Choose from k rank buckets by a power distribution and sample within
    buckets uniformally"""

    k = 32
    # pow_alpha = -0.7
    # pow_beta = -0.5
    
    # # P(i) = (1 / rank(i)) ^ alpha / sum((1 / rank(i)) ^ alpha)
    # pdf = [x ** pow_alpha for x in range(1, k+1)]
    # pdf_sum = math.fsum(pdf)
    # distribution = [x / pdf_sum for x in pdf]
    
    # # The expected value with stochastic updates depends on the behavior
    # # distribution to be the same as the updates. Since we are introducing
    # # bias with prioritized sweeps, we need to do introduce importance
    # # sampling.

    # # https://arxiv.org/pdf/1511.05952.pdf
    # # Importance sampling weight 
    # # w_i = (N * P(i))^-B) / max(w_j)
    # importance = [(len(timesteps) * pi) ** pow_beta for pi in distribution]
    # max_importance = max(importance)
    # importance = [i / max_importance for i in importance]
    
    # # Sample timesteps
    # # timesteps.sort(key=lambda t: t.abs_error, reverse=True)
    # samples = []
    # b = 0
    # for start, stop in batch_indexes(len(timesteps), math.floor(len(timesteps) / k)):
    #     if b == k: break
    #     sample = np.random.choice(timesteps[start:stop])
    #     sample.adjusted_target = sample.target * importance[b]
    #     samples.append(sample)
    #     b += 1
    
    for t in timesteps:
        t.adjusted_target = t.target
    random.shuffle(timesteps)
    return timesteps[:k]
    
def sum_TD_errors(timesteps):
    lamda = .7
    
    T = len(timesteps)
    for ind, t in enumerate(timesteps):
        t.update_static_val()

        error = 0
        L = 1
        for j in range(ind+2, T, 2):
            L *= lamda

            # target value comes from the target network for an unbaised evaluation
            t_next = timesteps[j]
            t_next.update_static_val()

            t_prev = timesteps[j-2]
            t_prev.update_static_val()

            delta = t_next.static_val - t_prev.static_val
            error += L * delta

        t.abs_error = abs(error)
        t.target = error + t.static_val
    
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

def validation_data(offset, npos):
    features = []
    scores = []
    with open("../stockfish_init_scores.txt") as initf:
        n = 0
        while True:
            fen = initf.readline()
            s_raw = initf.readline()
            if not s_raw: break
            s = int(s_raw)
            
            if offset < n <= offset + npos:
                s = math.tanh(s)
                scores.append(s)
                pos = Position.from_fen(fen)
                f = nn_evaluate.get_features(pos)
                features.append(f)

            if n >= offset + npos: break
            n += 1
    return features, scores
                
def initialize_weights_sf(npos):
    nvalid = 1e4
    ndata = npos - nvalid
    
    validation_f, validation_s = validation_data(offset=ndata, npos=nvalid)
    validation_s = [math.tanh(s / 190) for s in validation_s]
    features = []
    scores = []
    # fens = []
    with open("../stockfish_init_scores.txt") as initf:
        n = 0
        while n != ndata:
            fen = initf.readline()
            s = int(initf.readline())
            # s = min(1, max(-1, s / 1000))
            s = math.tanh(s / 190)
            scores.append(s)
            pos = Position.from_fen(fen)
            f = nn_evaluate.get_features(pos)
            features.append(f)
            # fens.append(fen)
            n += 1
    # print(len(fens), len(scores))
    print(len(features), len(scores))
    with open('../stockfish_init_validation.txt', 'a') as validf:
        k = 0
        for _ in range(10):
            # z = list(zip(fens, scores))
            z = list(zip(features, scores))
            random.shuffle(z)
            # rfens, rscores = list(zip(*z))
            rfeatures, rscores = list(zip(*z))
            # for start, stop in batch_indexes(len(rfens), 1024):
            for start, stop in batch_indexes(len(rfeatures), 1024):
                # fs = [nn_evaluate.get_features(Position.from_fen(fen)) for fen in rfens[start:stop]]
                # ss = rscores[start:stop]
                # model.actor.train_batch(fs, ss)
                model.actor.train_batch(rfeatures[start:stop], rscores[start:stop])
                if k % 10 == 0:
                    valid_loss = model.actor.loss(validation_f, validation_s)
                    validf.write("{}, ".format(valid_loss[0]))
                k += 1
            valid_loss = model.actor.loss(validation_f, validation_s)
            validf.write("{}, ".format(valid_loss[0]))
    model.copy_to_target()
    Timestep.current_target_generation += 1
    
def train(episodes, write_summary):
    features = []
    targets = []
    for timesteps in episodes:
        sum_TD_errors(timesteps)
        
    # # get random batch
    # z = list(zip(features, targets))
    # random.shuffle(z)
    # rfeatures, rtargets = list(zip(*z))
    # bfeatures, btargets = rfeatures[:batch_size], rtargets[:batch_size]

    # flatten the list
    timesteps = [t for e in episodes for t in e]
    # batch = sample(timesteps)
    
    epochs = 10
    for epoch in range(epochs):
        random.shuffle(timesteps)
        rfeatures, rtargets = zip(*[(t.features, t.target) for t in timesteps])
        for start, stop in batch_indexes(len(positions), 32):
            wr = write_summary and (epoch == (epochs-1)) and start == 0
            model.actor.train_batch(rfeatures[start:stop], rtargets[start:stop], write_summary=wr)
            
    # # update the actor
    # model.actor.train_batch(bfeatures, btargets, write_summary)


if __name__ == "__main__":
    depth = 5 #6.5
    total_fens = 700762
    plies_to_play = 32
    positions_per_iteration = 256
    # batch_size = 32 # plies_to_play * positions_per_iteration // 8
    num_iterations = total_fens // positions_per_iteration + 1
    max_replay_buffer_size = 64

    # init_npos = 295705
    init_npos = 200000
    offset = init_npos
    sts_scores = []
    # episodes = []
    for itern in range(num_iterations):
        episodes = []
        positions = []
        lines_read = 0
        # initialize_network = itern == 0 and not os.path.exists('checkpoint')
        initialize_network = True
        if initialize_network:
            npos = init_npos
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
            # initialize_weights(positions)
            initialize_weights_sf(npos)
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
                    
                    timesteps.append(Timestep(leaf=leaf,
                                              features=nn_evaluate.get_features(leaf),
                                              static_val=None,
                                              static_gen=-1,
                                              abs_error=None,
                                              target=None,
                                              adjusted_target=None))

                    # stop playing when the results are no longer the raw NN output
                    is_fixed = eval_is_fixed(leaf, leaf_val)
                    if is_fixed:
                        break
                        
                    psn.make_move(pv[0])
                    if len(timesteps) % 10 == 0:
                        print(psn)
                
                episodes.append(timesteps)
                
                # replay buffer size is limited
                # if len(episodes) > max_replay_buffer_size:
                #     episodes.pop(0)

                # # train after each episode rather than timestep because
                # # otherwise we'd have to turn off use of the transposition table
                # if len(episodes) == max_replay_buffer_size:
                #     write_summary = n % 32 == 0
                #     train(episodes, write_summary=write_summary)
                
            train(episodes, write_summary=True)
                
            # After each playing iteration:
            # .. check our progress
            sts_score = sts.run_sts_test()
            sts_scores.append(sts_score)
            model.update_sts_score(sts_score)
            with open('../sts_scores.txt', 'a') as f:
                f.write('{0}, '.format(sts_score))
            print("STS scores over time:", sts_scores)
            # .. update the target network and save the models
            model.copy_to_target()
            Timestep.current_target_generation += 1
            model.save_model()
