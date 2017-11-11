import pandas as pd
import numpy as np
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyximport
pyximport.install()
from bb import *
import nn_evaluate
from nn import model
from position import Position

# 4350 (1573elo) 323s orignn

plt.ion()
fig = plt.figure()
# plt.axis([], xmin=0, ymin=-1, ymax=1)

file_name = '../slonik_data/sf_scores.pkl'
sf_scores = pd.read_pickle(file_name)
batch_size = 32

valmod = 5000

train_n = int(len(sf_scores) - 6e4)

train_data = sf_scores[:train_n]
valid_data = sf_scores[train_n:]

print("Train data {} records".format(len(train_data)))
print("Valid data {} records".format(len(valid_data)))

epochs = 5

for epoch in range(epochs):

    train_loss = []
    valid_loss = []
    
    for iteration, batch in train_data.groupby(np.arange(len(train_data)) // batch_size):
        fens = [f.strip() for f in batch.fen.tolist()]
        feats = [nn_evaluate.get_features(Position.from_fen(fen)) for fen in fens]
        scores = np.array(batch.stockfish_score, dtype='float32')
        loss, _, _ = model.actor.train_batch(feats, scores)
        train_loss.append(loss)
        if iteration % 500 == 0:
            t_ = np.array(train_loss)
            progress = int((iteration + 1) / ((len(train_data) // batch_size) + 1) * 100)
            print("{}%, train loss: {:6.4f}, +/-{:6.4f}".format(progress, t_.mean(), t_.std()))
        # if iteration % valmod == 0:
    
    for valid_iteration, valid_batch in tqdm(valid_data.groupby(np.arange(len(valid_data)) // batch_size)):
        valid_fens = [f.strip() for f in batch.fen.tolist()]
        valid_feats = [nn_evaluate.get_features(Position.from_fen(fen)) for fen in valid_fens]
        valid_scores = np.array(batch.stockfish_score, dtype='float32')
        valid_loss += model.actor.loss(valid_feats, valid_scores)

    t = np.array(train_loss)
    v = np.array(valid_loss)
    print("train loss: {}, +/-{}".format(t.mean(), t.std()))
    print("valid loss: {}, +/-{}".format(v.mean(), v.std()))
    
    # fig.clear()
    # plt.plot(np.arange(len(train_loss)), train_loss, 'b')
    # plt.plot(np.arange(len(train_loss)), np.convolve(train_loss, np.ones(batch_size*10), mode='same') / batch_size*10)
    # plt.plot(np.arange(len(valid_loss)), valid_loss, 'g')
    # plt.show()
    # plt.pause(0.0001)

    model.copy_to_target()
    model.save_model()

pd.DataFrame(np.array(train_loss)).to_csv('../slonik_data/train_loss.csv')
pd.DataFrame(np.array(valid_loss)).to_csv('../slonik_data/valid_loss.csv')
