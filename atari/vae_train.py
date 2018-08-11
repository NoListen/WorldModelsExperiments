'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from utils import reset_graph, create_vae_dataset, check_dir, saveToFlat
import pickle
from config import env_name

# TODO decide train the VAE and RNN by alternation.

def load_data(load_flag, data_dir):
  if not os.path.exists("vae/data.p"):
    print("No data to load directly")
    load_flag = False

  if load_flag:
    with open("vae/data.p", "rb") as f:
      dataset = pickle.load(f)
  else:
    filelist = os.listdir(data_dir)
    filelist = [f for f in filelist if '.npz' in f]
    filelist = filelist[0:10000]
    dataset = create_vae_dataset(filelist, N=5000)
    #with open("vae/data.p", "wb") as f:
    #  pickle.dump(dataset, f, protocol=4)
  print("The dataset has shape", dataset.shape)
  return dataset

def learn(sess,z_size, batch_size, lr, kl_tolerance,
          num_epoch, model_dir, data_dir, load_flag=False):

  dataset = load_data(load_flag, data_dir)
  total_length = len(dataset)
  num_batches = int(np.floor(total_length/batch_size))
  ids = np.arange(total_length)
  np.random.shuffle(ids)
  print("num_batches", num_batches)

  check_dir(model_dir)
  print("the model will be saved to", model_dir)

  vae = ConvVAE(name="conv_vae",
                z_size=z_size,
                batch_size=batch_size)

  var_list = vae.get_variables()

  global_step = tf.Variable(0, name='global_step', trainable=False)
  # lr = tf.Variable(lr, name='learning_rate', trainable=False) # replace here

  # reconstruction loss
  tf_r_loss = -tf.reduce_sum(vae.x * tf.log(vae.y+1e-8) +
                                      (1.-vae.x) * (tf.log(1.-vae.y+1e-8)),[1,2,3])
  tf_r_loss = tf.reduce_mean(tf_r_loss)
  tf_kl_loss = - 0.5 * tf.reduce_sum(
    (1 + vae.logvar - tf.square(vae.mu) - tf.exp(vae.logvar)),
    axis=1)
  tf_kl_loss = tf.reduce_mean(tf.maximum(tf_kl_loss, kl_tolerance * z_size))
  tf_loss = tf_kl_loss + tf_r_loss

  # no decay
  optimizer = tf.train.AdamOptimizer(lr)
  tf_grads = optimizer.compute_gradients(tf_loss)  # can potentially clip gradients here.

  train_op = optimizer.apply_gradients(
    tf_grads, global_step=global_step, name='train_step')

  sess.run(tf.global_variables_initializer())
  # train loop:
  print("train", "step", "loss", "recon_loss", "kl_loss")
  for epoch in range(num_epoch):
    np.random.shuffle(ids)

    for i in range(num_batches):
      batch_id = ids[i*batch_size:(i+1)*batch_size]
      batch = dataset[batch_id]

      obs = batch.astype(np.float)/255.0
      feed = {vae.x: obs}

      (train_loss, r_loss, kl_loss, train_step, _) = sess.run([
        tf_loss, tf_r_loss, tf_kl_loss, global_step, train_op
      ], feed)

      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss, kl_loss)

      if ((train_step+1) % 5000 == 0):
        saveToFlat(var_list, model_dir+'/%i.p' % (train_step+1))

  saveToFlat(var_list, model_dir+'/final_vae.p')

def main():
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
  parser.add_argument('--batch-size', type=int, default=100, help="batch size")
  parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
  parser.add_argument('--kl-tolerance', type=float, default=0.5, help="kl tolerance")
  parser.add_argument('--num-epoch', type=int, default=10, help="number of training epochs")
  parser.add_argument('--data-dir', default="record", help="the directory of data")
  parser.add_argument('--model-dir', default="tf_vae", help="the directory to store vae model")
  parser.add_argument("--load-flag", action="store_true", default=False, help="load the data directly")
  args = vars(parser.parse_args())
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:
    learn(sess, **args)

if __name__ == '__main__':
  main()
