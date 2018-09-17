from mlp import MlpNetwork
from utils import *
import numpy as np
import tensorflow as tf


# input_size, output_size, hid_size, num_hid_layers


# TODO batch_size may be necessary
class Agent(object):
    def __init__(self, sess, lr, action_dict=None, policy_name="pi", **kargs):
        self.sess = sess
        self.net = MlpNetwork(name=policy_name, **kargs)
        self.action_dict = action_dict
        self.opt = tf.train.AdamOptimizer(learning_rate=lr,
                                beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.net.total_loss)

    def action(self, ob, stochastic=True):
        # obs is only one state by default
        feed_dict = {self.net.ob: ob[None]}
        prob, value = self.sess.run([self.net.p, self.net.v], feed_dict=feed_dict)
        # the batch_size is one
        prob = prob[0]
        value = value[0]
        # print("prob  " , prob)
        #print(prob)
        if not stochastic:
            action = np.argmax(prob, -1)
        else:
            action = np.random.choice(np.arange(len(prob)), p=prob)

        if self.action_dict:
            action = self.action_dict[action]
        return action, value

    # ep should have 1. ob 2. ret(discounted return) 3. action
    def train(self, seg):
        seg["ob"] = np.transpose(seg["ob"], [1, 0, 2])
        
        feed_dict = {self.net.ob: seg["ob"][0],
                     self.net.r: seg["tdlamret"],
                     self.net.action:seg["ac"],
                     self.net.td: seg["adv"]}

        self.sess.run(self.opt, feed_dict=feed_dict)

        #feed_dict = {self.net.ob: seg["ob"][1],
        #             self.net.r: seg["tdlamret"],
        #             self.net.action:seg["ac"],
        #             self.net.td: seg["adv"]}
        #self.sess.run(self.opt, feed_dict=feed_dict)
