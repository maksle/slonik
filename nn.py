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

def batch_indexes(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

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


class Model(object):
    def __init__(self, sess, graph, restore=False):
        self.sess = sess
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.graph = graph
        self.checkpoint_path = './'

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        global_step_op = self.global_step.assign_add(1)

        lamda = .7
        lr = .0005

        self.input_global_size, self.hidden_global_size = 26, 15
        self.input_pawn_size, self.hidden_pawn_size = 18, 14
        self.input_piece_size, self.hidden_piece_size = 68, 16
        self.input_square_size, self.hidden_square_size = 148, 16
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
        self.V_next = tf.placeholder('float', shape=(1,), name='V_next')
        self.target = tf.placeholder('float', shape=[None, 1], name='target')

        # fit_loss = tf.reduce_mean(self.target - self.V, name='fit_loss')
        self.fit_loss = tf.losses.absolute_difference(self.target, self.V)
        # fit_loss = tf.Print(fit_loss, [fit_loss], "Loss: ")
        self.fit_op = tf.train.AdamOptimizer(.0003).minimize(self.fit_loss)
        
        # stats
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            loss_sum_op = loss_sum.assign_add(loss_op)
            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            tf.summary.scalar('game/loss_avg', loss_avg_op)

        self.sts_score = tf.Variable(0.0, name='STS_score', trainable=False)
        tf.summary.scalar('test/sts_score', self.sts_score)
        
        delta_op = tf.reduce_mean(tf.abs(self.V_next - self.V))
        
        tvars = tf.trainable_variables()
        # grads = tf.gradients(self.V, tvars)
        grads = tf.gradients(self.V, tvars)
        
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)
        
        apply_gradients = []
        trace_reset_ops = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):

                with tf.variable_scope('trace'):
                    trace = tf.Variable(tf.zeros(grad.get_shape()), name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    tf.summary.histogram('trace', trace)

                    trace_reset_op = trace.assign(tf.zeros(grad.get_shape()))
                    trace_reset_ops.append(trace_reset_op)
                    
                grad_trace = lr * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                apply_gradients.append(var.assign_add(grad_trace))

        with tf.control_dependencies([
                global_step_op,
                loss_sum_op,
        ]):
            self.optimize = tf.group(*apply_gradients, name='optimize')

        # self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grad_and_vars)

        game_step_reset_op = game_step.assign(0.0)
        loss_sum_reset_op = loss_sum.assign(0.0)
        game_step_op = game_step.assign_add(1.0)
        reset_ops = trace_reset_ops + [game_step_reset_op, loss_sum_reset_op, game_step_op]
        self.reset_op = tf.group(*reset_ops, name='reset')
        
        self.sess.run(tf.global_variables_initializer())
        self.summaries_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('/tmp/td', self.sess.graph)
        self.saver = tf.train.Saver()
        if restore:
            self.restore()

    def save_model(self):
        # self.model.save('learnedmodel.h5')
        # with self.sess.as_default():
        self.saver.save(self.sess, self.checkpoint_path + 'tfcheckpoint') 
        print("saved model")
            
    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)
            
    def summary(self):
        self.model.summary()
    
    def transform_features(self, features):
        m = len(features)
        dgs = np.zeros((m, self.input_global_size))
        dpws = np.zeros((m, self.input_pawn_size))
        dpcs = np.zeros((m, self.input_piece_size))
        dsqs = np.zeros((m, self.input_square_size))
        for n, f in enumerate(features):
            dgs[n], dpws[n], dpcs[n], dsqs[n] = f
        return { 'input_global': dgs, 'input_pawn': dpws, 'input_piece': dpcs, 'input_square': dsqs }
        
    def fit(self, features, targets, epochs, batch_size):
        with self.graph.as_default():
            for epoch in range(epochs):
                c = list(zip(features, targets))
                random.shuffle(c)
                efeatures, etargets = list(zip(*c))
                tloss = 0
                for start, end in batch_indexes(len(etargets), batch_size):
                    f = self.transform_features(efeatures[start:end])
                    t = etargets[start:end]
                    loss, _ = self.sess.run([self.fit_loss, self.fit_op], feed_dict={
                        self.input_global: f['input_global'],
                        self.input_pawn: f['input_pawn'],
                        self.input_piece: f['input_piece'],
                        self.input_square: f['input_square'],
                        self.target: np.array(t).reshape(end-start, 1)
                    })
                    tloss += loss
                print("epoch", epoch, "loss", tloss)
                    
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

    def train(self, features, v_next):
        with self.graph.as_default():
            f = self.transform_features([features])
            _, summaries = self.sess.run([self.optimize, self.summaries_op], feed_dict={
                self.input_global: f['input_global'],
                self.input_pawn: f['input_pawn'],
                self.input_piece: f['input_piece'],
                self.input_square: f['input_square'],
                self.V_next: np.array([v_next]),
            })
            self.summary_writer.add_summary(summaries, global_step=self.global_step.eval(session=self.sess))

    def reset_eligibility_trace(self):
        self.sess.run(self.reset_op)

    def update_sts_score(self, score):
        self.sess.run(self.sts_score.assign(score))
        

graph = tf.Graph()
sess = tf.Session(graph=graph)
with sess.as_default(), graph.as_default():
    model = Model(sess, graph, restore=True)
