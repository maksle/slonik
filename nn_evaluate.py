import numpy as np
# from IPython import embed
from features import ToFeature

from nn import model # mlp

# from cnnfeat import model, get_feats #cnn
# import numpy as np


toFeature = ToFeature()

def get_features(position):
    toFeature.set_position(position)
    return toFeature.ann_features()

def evaluate(position, evaluator=model.actor): # mlp
# def evaluate(position, evaluator=model): # cnn

    # mlp
    features = get_features(position)
    res_value = evaluator.predict(np.expand_dims(features, 0))

    # cnn
    # features = get_feats(position)
    # res_value = evaluator.predict(np.expand_dims(features, axis=0), batch_size=1)

    if position.white_to_move():
        val = np.asscalar(res_value)
    else:
        val = -np.asscalar(res_value)
    return val * 1000
