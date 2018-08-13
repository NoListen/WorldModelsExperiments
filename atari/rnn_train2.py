'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import tensorflow as tf
import time
from tensorboard_logger import configure, log_value
from rnn.rnn import MDNRNN
from utils import saveToFlat, check_dir, tf_lognormal
from config import env_name
from env import make_env

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# TODO sample from the initial z
# initial_z_save_path = "tf_initial_z"
# if not os.path.exists(initial_z_save_path):
#   os.makedirs(initial_z_save_path)
# initial_mu = np.copy(data_mu[:1000, 0, :] * 10000).astype(np.int).tolist()
# initial_logvar = np.copy(data_logvar[:1000, 0, :] * 10000).astype(np.int).tolist()
# with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
#     json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

class DataSet(object):
    def __init__(self, filename):
        raw_data = np.load(filename)
        self.data_mu = raw_data["mu"]
        self.data_logvar = raw_data["logvar"]
        self.data_action =  raw_data["action"]
        self.max_seq_len = self.data_action.shape[1]-1
        print(type(raw_data))
        self.n_data = self.data_action.shape[0]
        self.ids = np.arange(self.n_data)
        self.i = 0
        np.random.shuffle(self.ids)

    def random_batch(self, batch_size):
        indices = self.ids[self.i:self.i + batch_size]
        nb = len(indices)
        mu = self.data_mu[indices]
        logvar = self.data_logvar[indices]
        a = self.data_action[indices]
        s = logvar.shape
        z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)
        if nb == 1:
            z = z[None]
            a = a[None]

        # z [N, z_size] action [N, na]
        self.i += batch_size
        if self.i >= self.n_data:
            np.random.shuffle(self.ids)
            self.i = 0

        # pad
        if nb < batch_size:
            tz, ta = self.random_batch(batch_size - nb)
            z = np.concatenate([z, tz], axis=0)
            a = np.concatenate([a, ta], axis=0)

        return z, a

def get_lossfunc(logmix, mean, logstd, y):
    v = logmix + tf_lognormal(y, mean, logstd)
    v = tf.reduce_logsumexp(v, 1, keepdims=True)
    return -tf.reduce_mean(v)

def learn(sess, z_size, data_dir, num_steps,
          batch_size=100, rnn_size=256,
          grad_clip=1.0, n_mix=3,
          lr=0.001, min_lr=0.00001, decay=0.99,
          model_dir="tf_rnn", layer_norm=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):

    check_dir(model_dir)
    configure("%s/%s_rnn" % (model_dir, env_name))

    # number of actions
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    dataset = DataSet(os.path.join(data_dir, "series.npz"))

    rnn = MDNRNN("rnn",
                 dataset.max_seq_len,
                 input_size,
                 output_size,
                 batch_size,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 n_mix,  # number of mixtures in MDN
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp)

    input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, dataset.max_seq_len, input_size])
    output_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, dataset.max_seq_len, output_size])

    out_logmix, out_mean, out_logstd, initial_state, final_state = rnn.build_model(input_x)
    # TODO Define Loss and other optimization stuff.
    print("Hello, we are going to ", output_x.shape, out_mean.shape)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    flat_target = tf.reshape(output_x, [-1, 1])

    loss = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target)
    loss = tf.reduce_mean(loss)

    tf_lr = tf.Variable(lr, trainable=False)
    optimizer = tf.train.AdamOptimizer(tf_lr)

    gvs = optimizer.compute_gradients(loss)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    # train optimizer
    train_op = optimizer.apply_gradients(clip_gvs, global_step=global_step, name='train_step')

    sess.run(tf.global_variables_initializer())
    curr_lr = lr
    # train loop:
    start = time.time()
    for local_step in range(num_steps):

      step = sess.run(global_step)
      curr_lr = (curr_lr-min_lr) * decay + min_lr

      raw_z, raw_a = dataset.random_batch(batch_size)
      inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
      outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions)

      feed = {input_x: inputs, output_x: outputs, tf_lr: curr_lr}
      (train_cost, state, train_step, _) = sess.run([loss, final_state, global_step, train_op], feed)
      if (step%20==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        log_value("training loss", train_cost, int(step//20))
        output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_lr, train_cost, time_taken)
        print(output_log)

    saveToFlat(rnn.get_variables(), model_dir+'/final_rnn.p')



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
    parser.add_argument('--data-dir', default="series", help="the directory of data")
    parser.add_argument('--num-steps', default=4000, help="number of training iterations")
    parser.add_argument('--batch-size', type=int, default=100, help="batch size")
    parser.add_argument('--rnn-size', type=int, default=256, help="rnn hidden state size")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="grad clip range")
    parser.add_argument('--n-mix', type=int, default=3, help="the number of gaussians in MDNRNN")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--min-lr', type=float, default=0.00001, help="minimum of learning rate")
    parser.add_argument('--decay', type=float, default=0.99999, help="decay of learning rate")

    parser.add_argument('--model-dir', default="tf_rnn", help="the directory to store rnn model")
    parser.add_argument('--layer-norm', action="store_true", default=False, help="layer norm in RNN")
    parser.add_argument('--recurrent-dp', type=float, default=1.0, help="dropout ratio in recurrent")
    parser.add_argument('--input-dp', type=float, default=1.0, help="dropout ratio in input")
    parser.add_argument('--output-dp', type=float, default=1.0, help="dropout ratio in output")

    args = vars(parser.parse_args())
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        learn(sess, **args)


if __name__ == '__main__':
  main()

