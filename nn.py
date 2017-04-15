from IPython import embed
import tensorflow as tf
import keras
import numpy as np
import random
import os
from keras.layers import Input, Dense
from keras.models import Model as KModel
import keras.backend as K


FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)


class Model(object):
    def __init__(self, sess, graph, restore=False):
        if os.path.exists('learnedmodel.h5'):
            self.model = keras.models.load_model('learnedmodel.h5')
        else:
            self.model = self.make_model()
        self.sess = sess
        self.graph = graph

        self.checkpoint_path = './'

        self.model.summary()
        
        lamda = .7
        lr = 1

        V = self.model.outputs

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        global_step_op = self.global_step.assign_add(1)
        
        tvars = tf.trainable_variables()
        grads = K.gradients(V, tvars)

        self.leaf = tf.placeholder('float', shape=(1,), name='leaf')
        self.leaf_next = tf.placeholder('float', shape=(1,), name='leaf_next')
        self.x = self.model.inputs

        loss_op = tf.reduce_mean(tf.square(self.leaf_next - self.leaf), name='loss')
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)
            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            loss_sum_op = loss_sum.assign_add(loss_op)
            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            tf.summary.scalar('game/loss_avg', loss_avg_op)
        
        delta_op = tf.reduce_sum(self.leaf_next - self.leaf)
        
        for grad, var in zip(grads, tvars):
            print(grad, var)
            if grad is None:
                continue
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)
        
        apply_gradients = []
        trace_reset_ops = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                if grad is None:
                    continue
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
                game_step_op,
                loss_sum_op,
        ]):
            self.optimize = tf.group(*apply_gradients, name='optimize')

        game_step_reset_op = game_step.assign(0.0)
        loss_sum_reset_op = loss_sum.assign(0.0)
        reset_ops = trace_reset_ops + [game_step_reset_op, loss_sum_reset_op]
        self.reset_op = tf.group(*reset_ops, name='reset')
            
        # self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grad_and_vars)
        self.sess.run(tf.global_variables_initializer())
        self.summaries_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('/tmp/td', self.sess.graph)
        self.saver = tf.train.Saver()
        if restore:
            self.restore()

    def save_model(self):
        # self.model.save('learnedmodel.h5')
        with self.sess.as_default():
            self.sess.run(self.global_step.assign(100))
            print(self.global_step.eval(session=self.sess))
            print("saved")
            self.saver.save(self.sess, self.checkpoint_path + 'tfcheckpoint') 
            
    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            # print("before restore", self.model.layers[4].get_weights()[0][0])
            self.saver.restore(self.sess, latest_checkpoint_path)
            print(self.global_step.eval(session=self.sess))
            embed()
            # print("after restore", self.model.layers[4].get_weights()[0][0])
                
    def make_model(self):
        input_global = Input(shape=(18,), name='input_global')
        hidden_global = Dense(18, activation='relu', name='hidden_global')(input_global)

        input_pawn = Input(shape=(18,), name='input_pawn')
        hidden_pawn = Dense(10, activation='relu', name='hidden_pawn')(input_pawn)

        input_piece = Input(shape=(68,), name='input_piece')
        hidden_piece = Dense(14, activation='relu', name='hidden_piece')(input_piece)

        input_square = Input(shape=(148,), name='input_square')
        hidden_square = Dense(14, activation='relu', name='hidden_square')(input_square)

        merged = keras.layers.concatenate([hidden_global, hidden_pawn, hidden_piece, hidden_square])
        hidden_shared = Dense(64, activation='relu', name='hidden_shared')(merged)

        output = Dense(1, activation='tanh', name='V')(hidden_shared)

        model = KModel(inputs=[input_global, input_pawn, input_piece, input_square], outputs=output)
        model.compile(optimizer='adam', loss='mean_absolute_error')

        return model
    
    def summary(self):
        self.model.summary()
    
    def transform_features(self, features):
        m = len(features)
        dgs = np.zeros((m, 18))
        dpws = np.zeros((m, 18))
        dpcs = np.zeros((m, 68))
        dsqs = np.zeros((m, 148))
        for n, f in enumerate(features):
            dgs[n], dpws[n], dpcs[n], dsqs[n] = f
        return { 'input_global': dgs, 'input_pawn': dpws, 'input_piece': dpcs, 'input_square': dsqs }
        
    def fit(self, features, targets, epochs, batch_size):
        with self.graph.as_default():
            f = self.transform_features(features)
            res = self.model.fit(f, np.asarray(targets), batch_size=batch_size, epochs=epochs)
            return np.asarray(res.history['loss']).mean()

    def predict(self, features):
        with self.sess.as_default(), self.graph.as_default():
            # embed()
            f = self.transform_features([features])
            return self.model.predict(f, verbose=0)[0][0]

    def train(self, features, v, v_next):
        f = self.transform_features([features])
        _, summaries = self.sess.run([self.optimize, self.summaries_op], feed_dict={
            self.leaf_next: np.array([v_next]),
            self.leaf: np.array([v]),
            self.model.get_layer('input_global').input: f['input_global'],
            self.model.get_layer('input_pawn').input: f['input_pawn'],
            self.model.get_layer('input_piece').input: f['input_piece'],
            self.model.get_layer('input_square').input: f['input_square']
        })
        self.summary_writer.add_summary(summaries, global_step=self.global_step.eval(session=self.sess))

    def reset_eligibility_trace(self):
        self.sess.run(self.reset_op)

graph = tf.Graph()
sess = tf.Session(graph=graph)
with sess.as_default(), graph.as_default():
    model = Model(sess, graph, restore=True)
