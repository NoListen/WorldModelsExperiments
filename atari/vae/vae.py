# ConvVAE model

import numpy as np
import json
import tensorflow as tf
import os

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

# TODO figure out the importance of reuse
# the first parameter must be name
class ConvVAE(object):
  def __init__(self, name, z_size=32, batch_size=100):
    self.name = name
    self.z_size= 32
    self.batch_size = 100

    # initialized
    with tf.variable_scope(name):
      self.scope = tf.get_variable_scope().name

  # Maybe
  # def update_scope(self):
  #   with tf.variable_scope(self.name):
  #     self.scope = tf.get_variable_scope().name

  # more like a call function
  def build_encoder(self, x, reuse=False):
    # it should be called in the scope where the instance is created.
    with tf.variable_scope(self.name):
      with tf.variable_scope("encoder", reuse=reuse):
        h = tf.layers.conv2d(x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
        h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
        h = tf.reshape(h, [-1, 2 * 2 * 256])

        # VAE
        self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
        self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
        sigma = tf.exp(self.logvar / 2.0)
        epsilon = tf.random_normal([self.batch_size, self.z_size])
        z = self.mu + sigma * epsilon
        return z

  def build_decoder(self, z):
    with tf.variable_scope(self.name):
      with tf.variable_scope("decoder", reuse=reuse):
        h = tf.layers.dense(z, 4*256, name="dec_fc")
        h = tf.reshape(h, [-1, 1, 1, 4*256])
        h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
        h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
        h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
        y = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
        return y

  def get_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


      # def get_random_model_params(self, stdev=0.5):
  #   # get random params.
  #   _, mshape, _ = self.get_model_params()
  #   rparam = []
  #   for s in mshape:
  #     #rparam.append(np.random.randn(*s)*stdev)
  #     rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
  #   return rparam
