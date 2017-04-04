from IPython import embed
import tensorflow as tf
import keras
import numpy as np
import random
import os
from keras.layers import Input, Dense
from keras.models import Model


tf.logging.set_verbosity(tf.logging.WARN)

cached_model = None

def make_model():
    input_global = Input(shape=(18,))
    hidden_global = Dense(18, activation='relu')(input_global)

    input_pawn = Input(shape=(18,))
    hidden_pawn = Dense(10, activation='relu')(input_pawn)

    input_piece = Input(shape=(68,))
    hidden_piece = Dense(14, activation='relu')(input_piece)

    input_square = Input(shape=(148,))
    hidden_square = Dense(14, activation='relu')(input_square)

    merged = keras.layers.concatenate([hidden_global, hidden_pawn, hidden_piece, hidden_square])
    hidden_shared = Dense(64, activation='relu')(merged)

    output = Dense(1, activation='tanh')(hidden_shared)

    model = Model(inputs=[input_global, input_pawn, input_piece, input_square], outputs=output)
    model.compile(optimizer='adam', loss='mean_absolute_error')

    model.summary()
    model.save('learnedmodel.h5')
    return model

def get_model():
    global cached_model
    if cached_model is None:
        if os.path.exists('learnedmodel.h5'):
            model = keras.models.load_model('learnedmodel.h5')
        else:
            model = make_model()
        cached_model = model
    else:
        model = cached_model
    return model
    
def transform_features(features):
    m = len(features)
    dgs = np.zeros((m, 18))
    dpws = np.zeros((m, 18))
    dpcs = np.zeros((m, 68))
    dsqs = np.zeros((m, 148))
    for n, f in enumerate(features):
        dgs[n], dpws[n], dpcs[n], dsqs[n] = f
    return [dgs, dpws, dpcs, dsqs]

def nn_fit(features, targets, epochs, batch_size):
    model = get_model()
    f = transform_features(features)
    res = model.fit(f, np.asarray(targets), batch_size=batch_size, epochs=epochs)
    return np.asarray(res.history['loss']).mean()

def nn_predict(features):
    model = get_model()
    f = transform_features([features])
    return model.predict(f, verbose=0)[0][0]

    

