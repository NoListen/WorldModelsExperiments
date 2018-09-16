import tensorflow as tf
import numpy as np

class MlpNetwork(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name):
            self._init(*args, **kargs)
            self.scope = tf.get_variable_scope().name
            self.cls = 'mlp'

    def _init(self, input_size, num_output, hid_size, num_hid_layers):
        assert len(input_size) == 1
        self.ob = tf.placeholder(tf.float32, shape=(None,) + input_size, name='obs')

        last_out = self.ob
        for i in range(num_hid_layers):
            last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size))

        # probabilities of all actions
        self.p = tf.nn.softmax(tf.layers.dense(last_out, num_output))

        # the value function
        self.v = tf.layers.dense(last_out, 1)

        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")
        # use Monte-Carlo Return at first.
        self.r = tf.placeholder(tf.float32, shape=(None,), name="return")
        self.td = tf.placeholder(tf.float32, shape=(None,), name="td")

        onehot_action = tf.one_hot(self.action, num_output, 1.0, 0.0, name="action_one_hot") # output_size = num_actions
        # log is the key idea in policy graident
        logp = tf.log(tf.clip_by_value(self.p, 1e-20, 1.0)) # avoid extreme situation
        logpa = tf.reduce_sum(tf.multiply(logp, onehot_action), axis=1)
        entropy = -tf.reduce_sum(self.p * logp, axis=1)

        self.policy_loss = -tf.reduce_sum(self.td * logpa + 0.01 * entropy)
        self.value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)
        self.total_loss = self.policy_loss + self.value_loss

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)