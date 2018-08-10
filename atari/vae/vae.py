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
  def __init__(self, name, *args, **kargs):
    with tf.variable_scope(name):
      self._init(*args, **kargs)
      self.scope = tf.get_variable_scope().name

  def _init(self, z_size=32, batch_size=100):
    self.z_size = z_size
    self.batch_size = batch_size
    self._build_graph()

  def _build_graph(self):
    #self.g = tf.Graph()
    #with self.g.as_default():

      # the input.
    self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])

      # Encoder
    h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
    h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
    h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
    h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
    h = tf.reshape(h, [-1, 2*2*256])

      # VAE
    self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
    self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
    self.sigma = tf.exp(self.logvar / 2.0)
    self.epsilon = tf.random_normal([self.batch_size, self.z_size])
    self.z = self.mu + self.sigma * self.epsilon

      # Decoder
    h = tf.layers.dense(self.z, 4*256, name="dec_fc")
    h = tf.reshape(h, [-1, 1, 1, 4*256])
    h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
    h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
    h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
    self.y = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")


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
