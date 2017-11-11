import pyximport
pyximport.install()
from IPython import embed
from bb import *
from piece_type import PieceType as Pt
from move_gen import *
from side import Side as S
import numpy as np
from position import Position
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import *
import keras
from keras.optimizers import Adam
from math import ceil
import pandas as pd

def bitfield(bitboard):
    return np.array([1 if b == '1' else 0 for b in bin(bitboard)[2:].zfill(64)]).reshape((8,8))

def get_feats(pos):
    krights = 0
    if preserved_kingside_castle_rights(pos.position_flags, Side.W):
        krights |= G1 | F1
    if preserved_queenside_castle_rights(pos.position_flags, Side.W):
        krights |= C1 | D1
    if preserved_kingside_castle_rights(pos.position_flags, Side.B):
        krights |= G8 | F8
    if preserved_kingside_castle_rights(pos.position_flags, Side.B):
        krights |= C8 | D8
    krights_plane = bitfield(krights)
    stm_plane = np.zeros(64).reshape((8,8)) if pos.side_to_move() == S.W else np.ones(64).reshape((8,8))
    wp_plane = bitfield(pos.pieces[Pt.piece(Pt.P, S.W)])
    bp_plane = bitfield(pos.pieces[Pt.piece(Pt.P, S.B)])
    wn_plane = bitfield(pos.pieces[Pt.piece(Pt.N, S.W)])
    bn_plane = bitfield(pos.pieces[Pt.piece(Pt.N, S.B)])
    wb_plane = bitfield(pos.pieces[Pt.piece(Pt.B, S.W)])
    bb_plane = bitfield(pos.pieces[Pt.piece(Pt.B, S.B)])
    wr_plane = bitfield(pos.pieces[Pt.piece(Pt.R, S.W)])
    br_plane = bitfield(pos.pieces[Pt.piece(Pt.R, S.B)])
    wq_plane = bitfield(pos.pieces[Pt.piece(Pt.Q, S.W)])
    bq_plane = bitfield(pos.pieces[Pt.piece(Pt.Q, S.B)])
    wk_plane = bitfield(pos.pieces[Pt.piece(Pt.K, S.W)])
    bk_plane = bitfield(pos.pieces[Pt.piece(Pt.K, S.B)])
    return np.stack([krights_plane, stm_plane, wp_plane, bp_plane, wn_plane, bn_plane, wb_plane, bb_plane,
                     wr_plane, br_plane, wq_plane, bq_plane, wk_plane, bk_plane], axis=-1) # 8x8x14

def conv_block(input_tensor, filters, kernel_size=(3,3), act=True):
    # embed()
    x = Convolution2D(filters, kernel_size, strides=(1,1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    if act:
        x = Activation('relu')(x)
    return x

def res_block(input_tensor, filters):
    x = conv_block(input_tensor, filters)
    # x = conv_block(x, filters)
    x = conv_block(x, filters, act=False)
    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# huber = tf.losses.huber

def batch_generator(data, batch_size, shuffle=True):
    while True:
        data_ = data.sample(frac=1) if shuffle else data
        for iteration, batch in data_.groupby(np.arange(len(data)) // batch_size):
            fens = [f.strip() for f in batch.fen.tolist()]
            feats = np.stack([get_feats(Position.from_fen(fen)) for fen in fens])
            scores = np.array(batch.stockfish_score, dtype='float32')
            yield feats, scores

def cnn_model():
    k = 192
    net_input = Input((8,8,14))
    x = conv_block(net_input, k, kernel_size=(3,3))
    for i in range(8):
        x = res_block(x, k)
    x = conv_block(x, 1, kernel_size=(1,1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    net_output = Activation('tanh')(x)
    return net_input, net_output

def dense_model():
    input_global = Input((26,))
    input_pawn = Input((18,))
    input_piece = Input((68,))
    input_square = Input((148,))
    g = Dense(26)(input_global)
    g = Activation('relu')(g)
    pwn = Dense(18)(input_pawn)
    pwn = Activation('relu')(pwn)
    pc = Dense(24)(input_piece)
    pc = Activation('relu')(pc)
    sq = Dense(32)(input_square)
    sq = Activation('relu')(sq)
    merged = keras.layers.concatenate([g, pwn, pc, sq])
    shared = Dense(64)(merged)
    shared = Activation('relu')(shared)
    output = Dense(1)(shared)
    output = Activation('tanh')(output)
    return [input_global, input_pawn, input_piece, input_square], output

inputs, output = dense_model()
model = Model(inputs, output)
model.summary()

# net_input, net_output = cnn_model()
# model = Model(net_input, net_output)
# # model.summary()
# model.compile(optimizer=Adam(1e-3), loss='mse')
# model.load_weights('../slonik_data/cnn_weights_epoch3.h5')

def train_data_supervised():
    file_name = '../slonik_data/sf_scores.pkl'
    sf_scores = pd.read_pickle(file_name)
    valid_n = 6e4
    train_n = int(len(sf_scores) - valid_n)
    train_data = sf_scores[:train_n]
    valid_data = sf_scores[train_n:]
    return train_data, valid_data

def train_supervised(batch_size=32, epochs=1):
    train_data, valid_data = train_data_supervised()
    
    train_gen = batch_generator(train_data, batch_size)
    valid_gen = batch_generator(valid_data, batch_size, shuffle=False)
    
    train_steps = ceil(len(train_data) / batch_size)
    valid_steps = ceil(valid_n / batch_size)
    
    # keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    model.fit_generator(train_gen, train_steps, epochs=epochs, validation_data=valid_gen, validation_steps=valid_steps)
    # model.save_weights('../slonik_data/cnn_weights.h5')
    
def evaluate():
    # .1256964 untrained
    # .0028118 after 1 epoch (1e-3 lr)
    # .0015 after 2 epoch (1e-3 lr)
    # 7.5081e-04 after 3 epoch (1e-4 lr)
    train_data, valid_data = train_data_supervised()
    valid_steps = ceil(valid_n / batch_size)
    valid_gen = batch_generator(valid_data, batch_size, shuffle=False)
    return model.evaluate_generator(valid_gen, valid_n//batch_size)
