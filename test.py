from IPython import embed
import tensorflow as tf
import gc
from nn import model


# saver = tf.train.import_meta_graph("./tfcheckpoint.meta")
# sess = tf.Session()
# saver.restore(sess, "./tfcheckpoint")
# with sess:
with model.sess:
    # coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tvars = tf.trainable_variables()
    for v in tvars:
        # if v.name == 'global_step:0':
        #     print(v.eval())
        print(v.name, v)
    gc.collect()


# [('hidden_global/kernel:0',
#   <tf.Tensor 'hidden_global/random_uniform:0' shape=(18, 18) dtype=float32>),
#  ('hidden_global/bias:0',
#   <tf.Tensor 'hidden_global/Const:0' shape=(18,) dtype=float32>),
#  ('hidden_pawn/kernel:0',
#   <tf.Tensor 'hidden_pawn/random_uniform:0' shape=(18, 10) dtype=float32>),
#  ('hidden_pawn/bias:0',
#   <tf.Tensor 'hidden_pawn/Const:0' shape=(10,) dtype=float32>),
#  ('hidden_piece/kernel:0',
#   <tf.Tensor 'hidden_piece/random_uniform:0' shape=(68, 14) dtype=float32>),
#  ('hidden_piece/bias:0',
#   <tf.Tensor 'hidden_piece/Const:0' shape=(14,) dtype=float32>),
#  ('hidden_square/kernel:0',
#   <tf.Tensor 'hidden_square/random_uniform:0' shape=(148, 14) dtype=float32>),
#  ('hidden_square/bias:0',
#   <tf.Tensor 'hidden_square/Const:0' shape=(14,) dtype=float32>),
#  ('hidden_shared/kernel:0',
#   <tf.Tensor 'hidden_shared/random_uniform:0' shape=(56, 64) dtype=float32>),
#  ('hidden_shared/bias:0',
#   <tf.Tensor 'hidden_shared/Const:0' shape=(64,) dtype=float32>),
#  ('V/kernel:0', <tf.Tensor 'V/random_uniform:0' shape=(64, 1) dtype=float32>),
#  ('V/bias:0', <tf.Tensor 'V/Const:0' shape=(1,) dtype=float32>),
#  ('iterations:0',
#   <tf.Tensor 'iterations/initial_value:0' shape=() dtype=float32>),
#  ('lr:0', <tf.Tensor 'lr/initial_value:0' shape=() dtype=float32>),
#  ('beta_1:0', <tf.Tensor 'beta_1/initial_value:0' shape=() dtype=float32>),
#  ('beta_2:0', <tf.Tensor 'beta_2/initial_value:0' shape=() dtype=float32>),
#  ('decay:0', <tf.Tensor 'decay/initial_value:0' shape=() dtype=float32>),
#  ('apply_gradients/trace/trace:0',
#   <tf.Tensor 'apply_gradients/trace/zeros:0' shape=(18, 18) dtype=float32>),
#  ('apply_gradients/trace_1/trace:0',
#   <tf.Tensor 'apply_gradients/trace_1/zeros:0' shape=(18,) dtype=float32>),
#  ('apply_gradients/trace_2/trace:0',
#   <tf.Tensor 'apply_gradients/trace_2/zeros:0' shape=(18, 10) dtype=float32>),
#  ('apply_gradients/trace_3/trace:0',
#   <tf.Tensor 'apply_gradients/trace_3/zeros:0' shape=(10,) dtype=float32>),
#  ('apply_gradients/trace_4/trace:0',
#   <tf.Tensor 'apply_gradients/trace_4/zeros:0' shape=(68, 14) dtype=float32>),
#  ('apply_gradients/trace_5/trace:0',
#   <tf.Tensor 'apply_gradients/trace_5/zeros:0' shape=(14,) dtype=float32>),
#  ('apply_gradients/trace_6/trace:0',
#   <tf.Tensor 'apply_gradients/trace_6/zeros:0' shape=(148, 14) dtype=float32>),
#  ('apply_gradients/trace_7/trace:0',
#   <tf.Tensor 'apply_gradients/trace_7/zeros:0' shape=(14,) dtype=float32>),
#  ('apply_gradients/trace_8/trace:0',
#   <tf.Tensor 'apply_gradients/trace_8/zeros:0' shape=(56, 64) dtype=float32>),
#  ('apply_gradients/trace_9/trace:0',
#   <tf.Tensor 'apply_gradients/trace_9/zeros:0' shape=(64,) dtype=float32>),
#  ('apply_gradients/trace_10/trace:0',
#   <tf.Tensor 'apply_gradients/trace_10/zeros:0' shape=(64, 1) dtype=float32>),
#  ('apply_gradients/trace_11/trace:0',
#   <tf.Tensor 'apply_gradients/trace_11/zeros:0' shape=(1,) dtype=float32>)]
