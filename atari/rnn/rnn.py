import numpy as np
from collections import namedtuple
import json
import tensorflow as tf

# hyperparameters for our model. I was using an older tf version, when HParams was not available ...

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3  # extra hidden later
MODE_ZH = 4

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)


# MDN-RNN model
class MDNRNN():
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self._init(*args, **kargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, *args， **kargs):
        self._build_graph(*args, **kargs)

    def _build_graph(self，num_steps,
                 max_seq_len,
                 input_size,
                 output_size,
                 batch_size,  # minibatch sizes
                 rnn_size=256,  # number of rnn cells
                 grad_clip=1.0,
                 num_mixture=3,  # number of mixtures in MDN
                 learning_rate=0.001,
                 decay_rate=1.0,
                 min_learning_rate=0.00001,
                 use_layer_norm=False,
                 use_recurrent_dropout=False,
                 recurrent_dropout_prob=0.90,
                 use_input_dropout=False,
                 input_dropout_prob=0.90,
                 use_output_dropout= False,
                 output_dropout_prob=0.90,
                 is_training=True):

        self.n_mix = num_mixture
        self.input_size = input_size  # 35 channels
        self.output_size = output_size  # 32 channels
        # TODO apply dynamic RNN
        self.max_seq_len = max_seq_len  # 1000 timesteps

        if is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell  # use LayerNormLSTM

        if use_recurrent_dropout:
            cell = cell_fn(rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(rnn_size, layer_norm=use_layer_norm)

        # multi-layer, and dropout:
        print("input dropout mode =", use_input_dropout)
        print("output dropout mode =", use_output_dropout)
        print("recurrent dropout mode =", use_recurrent_dropout)
        if use_input_dropout:
            print("applying dropout to input with keep_prob =", input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=input_dropout_prob)
        if use_output_dropout:
            print("applying dropout to output with keep_prob =", output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_dropout_prob)
        self.cell = cell

        self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, input_size])
        self.output_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, output_size])

        actual_input_x = self.input_x
        self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        # the output size
        n_out = output_size * num_mixture * 3

        with tf.variable_scope('RNN'):
            output_w = tf.get_variable("output_w", [rnn_size, n_out])
            output_b = tf.get_variable("output_b", [n_out])

        output, last_state = tf.nn.dynamic_rnn(cell, actual_input_x, initial_state=self.initial_state,
                                               time_major=False, swap_memory=True, dtype=tf.float32, scope="RNN")

        output = tf.reshape(output, [-1, rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        output = tf.reshape(output, [-1, num_mixture * 3]) # mean std weight
        self.final_state = last_state

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

        def get_lossfunc(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.reduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)

        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd

        out_logmix, out_mean, out_logstd = get_mdn_coef(output)

        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.output_x, [-1, 1])

        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)

        self.cost = tf.reduce_mean(lossfunc)

        if is_training:
            self.lr = tf.Variable(learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

        # initialize vars
        self.init = tf.global_variables_initializer()

    # def get_random_model_params(self, stdev=0.5):
    #     # get random params.
    #     _, mshape, _ = self.get_model_params()
    #     rparam = []
    #     for s in mshape:
    #         # rparam.append(np.random.randn(*s)*stdev)
    #         rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
    #     return rparam
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


def get_pi_idx(x, pdf):
    # samples from a categorial distribution
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print('error with sampling ensemble')
    return -1


# def rnn_init_state(rnn):
#     return rnn.sess.run(rnn.initial_state)
#
#
# def rnn_next_state(rnn, z, a, prev_state):
#     input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, -1))), axis=2)
#     feed = {rnn.input_x: input_x, rnn.initial_state: prev_state}
#     return rnn.sess.run(rnn.final_state, feed)
#
#
# def rnn_output_size(mode):
#     if mode == MODE_ZCH:
#         return (32 + 256 + 256)
#     if (mode == MODE_ZC) or (mode == MODE_ZH):
#         return (32 + 256)
#     return 32  # MODE_Z or MODE_Z_HIDDEN
#
#
# def rnn_output(state, z, mode):
#     if mode == MODE_ZCH:
#         return np.concatenate([z, np.concatenate((state.c, state.h), axis=1)[0]])
#     if mode == MODE_ZC:
#         return np.concatenate([z, state.c[0]])
#     if mode == MODE_ZH:
#         return np.concatenate([z, state.h[0]])
#     return z  # MODE_Z or MODE_Z_HIDDEN