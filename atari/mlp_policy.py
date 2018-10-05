import tensorflow as tf
from distributions import *
import tf_utils as U
import numpy as np


class MlpPolicy(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(*args, **kargs)


    def _init(self, input_size, num_output, hid_size, num_hid_layers):
        assert len(input_size) == 1
        # the number of actions
        self.pdtype = CategoricalPdType(num_output) # FIXED

        self.ob = tf.placeholder(tf.float32, shape=(None,) + input_size, name='obs')

        last_out = self.ob
        for i in range(num_hid_layers):
            last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size))

        pdparam = tf.contrib.layers.fully_connected(last_out, num_output,
                                                    activation_fn=tf.nn.relu, scope='policy_final')
        self.pd = self.pdtype.pdfromflat(pdparam)

        # select or sample actions
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        self.vpred = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None, scope='value_final')
        self._act = U.function([self.stochastic, self.ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred = self._act(stochastic, ob[None])
        return ac1[0], vpred[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
