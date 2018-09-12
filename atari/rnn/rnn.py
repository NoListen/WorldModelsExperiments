import numpy as np
from collections import namedtuple
import json
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# TODO consider controller
# Now focus on RNN & VAE
# MODE_ZCH = 0
# MODE_ZC = 1
# MODE_ZH = 4
# MODE_Z = 2

class VariantRNNCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    def __init__(self, var_dict, *args, **kargs):
        super(VariantRNNCell, self).__init__(*args, **kargs)
        self.var_dict = var_dict
        # just reuse the layer_norm
        # but get a different kernel.

    def _linear(self, args):
        weights = self.var_dict['kernel']
        out = math_ops.matmul(args, weights)
        if not self._layer_norm:
            bias = self.var_dict['bias']
            out = nn_ops.bias_add(out, bias)
        return out

# MDN-RNN model
class MDNRNN():
    def __init__(self, name,
                 max_seq_len,
                 input_size,
                 output_size,
                 batch_size,  # minibatch sizes
                 rnn_size=256,  # number of rnn cells
                 num_mixture=3,  # number of mixtures in MDN
                 layer_norm=False,
                 recurrent_dropout_prob=1.0,
                 input_dropout_prob=1.0,
                 output_dropout_prob=1.0):

        # these parameters determine the architecutre of RNN
        self.name = name
        self.max_seq_len = max_seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.n_mix = num_mixture
        self.layer_norm = layer_norm
        self.recurrent_dp = recurrent_dropout_prob
        self.input_dp = input_dropout_prob
        self.output_dp = output_dropout_prob

        print("input dropout mode =", input_dropout_prob < 1.0)
        print("output dropout mode =", output_dropout_prob < 1.0)
        print("recurrent dropout mode =", recurrent_dropout_prob < 1.0)

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

    def build_model(self, input_x, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell  # use LayerNormLSTM

            if self.recurrent_dp < 1.0:
                cell = cell_fn(self.rnn_size, layer_norm=self.layer_norm,
                               dropout_keep_prob=self.recurrent_dp)
            else:
                cell = cell_fn(self.rnn_size, layer_norm=self.layer_norm)

            if self.input_dp < 1.0:
                print("applying dropout to input with keep_prob =", self.input_dp)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.input_dp)

            if self.output_dp < 1.0:
                print("applying dropout to output with keep_prob =", self.output_dp)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_dp)

            n_out = self.output_size * self.n_mix * 3
            output_w = tf.get_variable("output_w", [self.rnn_size, n_out])
            output_b = tf.get_variable("output_b", [n_out])

            return self.build_base_model(cell, input_x, output_w, output_b)

    def build_variant_model(self, input_x, weight_dict, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            cell_fn = VariantRNNCell  # use LayerNormLSTM

            if self.recurrent_dp < 1.0:
                cell = cell_fn(weight_dict, self.rnn_size, layer_norm=self.layer_norm,
                               dropout_keep_prob=self.recurrent_dp)
            else:
                cell = cell_fn(weight_dict, self.rnn_size, layer_norm=self.layer_norm)

            
            if self.input_dp < 1.0:
                print("applying dropout to input with keep_prob =", self.input_dp)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.input_dp)

            if self.output_dp < 1.0:
                print("applying dropout to output with keep_prob =", self.output_dp)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_dp)
            
            output_w = weight_dict["output_w"]
            output_b = weight_dict["output_b"]

            return self.build_base_model(cell, input_x, output_w, output_b)

    def build_base_model(self, cell, input_x, output_w, output_b):
            input_x_list = tf.split(input_x, self.max_seq_len, axis=1)
            input_x_list = [tf.squeeze(i) for i in input_x_list]
            # TODO make use of initial state
            # initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            output, last_state = tf.nn.static_rnn(cell, input_x_list, #initial_state=initial_state,
                                                   dtype=tf.float32, scope="RNN")
            output = tf.stack(output, axis=1)
            output = tf.reshape(output, [-1, self.rnn_size])
            output = tf.nn.xw_plus_b(output, output_w, output_b)
            output = tf.reshape(output, [-1, self.n_mix * 3]) # mean std weight
            final_state = last_state

            def get_mdn_coef(output):
                logmix, mean, logstd = tf.split(output, 3, 1)
                logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
                return logmix, mean, logstd

            out_logmix, out_mean, out_logstd = get_mdn_coef(output)

            return out_logmix, out_mean, out_logstd, final_state


    # def get_random_model_params(self, stdev=0.5):
    #     # get random params.
    #     _, mshape, _ = self.get_model_params()
    #     rparam = []
    #     for s in mshape:
    #         # rparam.append(np.random.randn(*s)*stdev)
    #         rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
    #     return rparam
    def get_linear_variables(self):
        # return a dictionary
        lv_dict = {}
        v_list = self.get_variables()
        for var in v_list:
            v_type = var.name.split('/')[-1].split(':')[0]
            if v_type == "kernel":
                lv_dict["kernel"] = var
            elif v_type == "bias":
                lv_dict["bias"] = var
            elif v_type == "output_w":
                lv_dict["output_w"] = var
            elif v_type == "output_b":
                lv_dict["output_b"] = var
            else:
                continue
            print(var.name, "has been added to linear variables")
        return lv_dict

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
