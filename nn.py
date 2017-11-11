from IPython import embed
import tensorflow as tf
# import keras
import numpy as np
import random
import os
# from keras.layers import Input, Dense
# from keras.models import Model as KModel
# import keras.backend as K

# from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)

# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b

# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


class Estimator(object):
    def __init__(self, sess, graph, scope, global_step):
        self.sess = sess
        self.graph = graph
        self.scope = scope
        self.global_step = global_step

        with tf.variable_scope(scope):
            self.build_model()
        
        summary_dir = os.path.join('/tmp/td', 'summaries_{}'.format(scope))
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        # self.summary_writer = tf.train.SummaryWriter(summary_dir)

    def build_model(self):
        self.input_global_size, self.hidden_global_size = 26, 26
        self.input_pawn_size, self.hidden_pawn_size = 18, 18
        self.input_piece_size, self.hidden_piece_size = 68, 24
        self.input_square_size, self.hidden_square_size = 148, 32
        merged_size = self.hidden_global_size + self.hidden_pawn_size + self.hidden_piece_size + self.hidden_square_size
        hidden_shared_size = 64

        # build the graph
        # .. input layer groups
        self.input_global = tf.placeholder('float', [None, self.input_global_size], name='input_global')
        self.input_pawn = tf.placeholder('float', [None, self.input_pawn_size], name='input_pawn')
        self.input_piece = tf.placeholder('float', [None, self.input_piece_size], name='input_piece')
        self.input_square = tf.placeholder('float', [None, self.input_square_size], name='input_square')
        # .. hidden layer groups
        hidden_global = dense_layer(self.input_global, [self.input_global_size, self.hidden_global_size], tf.nn.relu, name='hidden_global')
        hidden_pawn = dense_layer(self.input_pawn, [self.input_pawn_size, self.hidden_pawn_size], tf.nn.relu, name='hidden_pawn')
        hidden_piece = dense_layer(self.input_piece, [self.input_piece_size, self.hidden_piece_size], tf.nn.relu, name='hidden_piece')
        hidden_square = dense_layer(self.input_square, [self.input_square_size, self.hidden_square_size], tf.nn.relu, name='hidden_square')
        merged = tf.concat([hidden_global, hidden_pawn, hidden_piece, hidden_square], axis=1)
        # .. merged hidden layer
        hidden_shared = dense_layer(merged, [merged_size, hidden_shared_size], tf.nn.relu, name='hidden_shared')
        # .. output layer
        self.V = dense_layer(hidden_shared, [hidden_shared_size, 1], tf.tanh, name='V')
        
        # placeholders for training
        self.target = tf.placeholder('float', shape=[None, 1], name='target')
        
        
        with tf.name_scope('Loss'):
            self.fit_loss = tf.reduce_mean(huber_loss(self.target, self.V))
            tf.summary.scalar('loss', self.fit_loss)
        
        with tf.name_scope('SGD'):
            tvars = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
            grads = tf.gradients(self.fit_loss, tvars)
            for grad, var in zip(grads, tvars):
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(var.name + '/gradients/grad', grad)
            # grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
            # optimizer = tf.train.AdamOptimizer(.0003)
            # self.fit_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self.fit_op = tf.train.AdamOptimizer(1e-5).minimize(self.fit_loss, global_step=self.global_step)
            
        self.sts_score = tf.Variable(0.0, name='sts_score', trainable=False)
        tf.summary.scalar('sts_score', self.sts_score)
        
        # summaries
        self.summaries_op = tf.summary.merge_all()
        
    def predict(self, features):
        with self.graph.as_default():
            f = self.transform_features([features])
            v = self.sess.run([self.V], feed_dict={
                self.input_global: f['input_global'],
                self.input_pawn: f['input_pawn'],
                self.input_piece: f['input_piece'],
                self.input_square: f['input_square']
            })
            return np.asscalar(v[0])

    def loss(self, features, targets):
        with self.graph.as_default():
            f = self.transform_features(features)
            loss = self.sess.run([self.fit_loss], feed_dict={
                self.input_global: f['input_global'],
                self.input_pawn: f['input_pawn'],
                self.input_piece: f['input_piece'],
                self.input_square: f['input_square'],
                self.target: np.array(targets).reshape(len(targets), 1)
            })
            return loss
            
    def train_batch(self, features, targets, write_summary=True):
        with self.graph.as_default():
            f = self.transform_features(features)
            ops = [self.fit_loss, self.fit_op]
            if write_summary:
                ops.append(self.summaries_op)
            results = self.sess.run(ops, feed_dict={
                self.input_global: f['input_global'],
                self.input_pawn: f['input_pawn'],
                self.input_piece: f['input_piece'],
                self.input_square: f['input_square'],
                self.target: np.array(targets).reshape(len(targets), 1)
            })
            if write_summary:
                self.summary_writer.add_summary(results[2], global_step=self.global_step.eval(self.sess))
            
    def transform_features(self, features):
        m = len(features)
        dgs = np.zeros((m, self.input_global_size))
        dpws = np.zeros((m, self.input_pawn_size))
        dpcs = np.zeros((m, self.input_piece_size))
        dsqs = np.zeros((m, self.input_square_size))
        for n, f in enumerate(features):
            dgs[n], dpws[n], dpcs[n], dsqs[n] = f
        return { 'input_global': dgs, 'input_pawn': dpws, 'input_piece': dpcs, 'input_square': dsqs }

    def update_sts_score(self, score):
        self.sess.run([self.sts_score.assign(score)])
        
class Model(object):
    def __init__(self, sess, graph, restore=False):
        self.sess = sess
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.graph = graph
        self.checkpoint_path = './'

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        self.sts_score = tf.summary.scalar('sts_score', 0.0)
        
        # self.summaries_op = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter('/tmp/td', self.sess.graph)

        self.actor = Estimator(self.sess, self.graph, "actor_estimator", self.global_step)
        self.target = Estimator(self.sess, self.graph, "target_estimator", self.global_step)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        if restore:
            self.restore()
            
    def save_model(self):
        with self.graph.as_default():
            self.saver.save(self.sess, self.checkpoint_path + 'tfcheckpoint') 
            print("saved model")
            
    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)
            
    def copy_to_target(self):
        with self.graph.as_default():
            e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.actor.scope)]
            e1_params = sorted(e1_params, key=lambda v: v.name)
            e2_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target.scope)]
            e2_params = sorted(e2_params, key=lambda v: v.name)

            update_ops = []
            for e1_v, e2_v in zip(e1_params, e2_params):
                op = e2_v.assign(e1_v)
                update_ops.append(op)

            self.sess.run(update_ops)
    
    def update_sts_score(self, score):
        self.actor.update_sts_score(score)
        

graph = tf.Graph()
# print("graphid:", id(graph))
sess = tf.Session(graph=graph)
with sess.as_default(), graph.as_default():
    model = Model(sess, graph, restore=True)
