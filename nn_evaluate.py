# from IPython import embed
from features import ToFeature
from nn import model


toFeature = ToFeature()

def get_features(position):
    toFeature.set_position(position)
    return toFeature.ann_features()

def evaluate(position, train=False):
    features = get_features(position)
    # embed()
    res_value = model.predict(features)
    if position.white_to_move():
        val = res_value
    else:
        val = -res_value
    # if abs(val * 1000 - 18.847251310944557) < .0008:
    #     embed()
    return val if train else val * 1000
