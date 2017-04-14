from features import ToFeature
from nn import model


toFeature = ToFeature()

def get_features(position):
    toFeature.set_position(position)
    return toFeature.ann_features()

def evaluate(position, train=False):
    features = get_features(position)
    res_value = model.predict(features)
    if position.white_to_move():
        val = res_value
    else:
        val = -res_value
    return val if train else val * 1000
