# from keras import backend as K
# from keras.models import Model, Sequential

import pandas as pd
# from bcolz_array_iterator import BcolzArrayIterator
import numpy as np
from nn import model
import os

file_name = '../slonik_data/sf_scores.pkl'

if os.path.exists(file_name):
    data = pd.read_pickle(file_name)
else:
    data = pd.DataFrame(columns=('fen', 'stockfish_score'))

# arr = bcolz.carray(np.empty(0, ))

for f in [# '../slonik_data/stockfish_init_scores_1.txt',
          # '../slonik_data/stockfish_init_scores_2.txt',
          '../slonik_data/stockfish_init_scores_3.txt'
]:
    with open(f) as scores:
        print('Processing {}'.format(f))
        n = 0
        while True:
            fen = scores.readline()
            score = scores.readline()
            if not score: break
            
            data.loc[data.shape[0]] = [fen.strip(), score.strip()]
            n += 1
            if n % 5000 == 0:
                print('saving', n)
                data.to_pickle(file_name)
        data.to_pickle(file_name)
    
data.to_pickle(file_name)
print("Done")
