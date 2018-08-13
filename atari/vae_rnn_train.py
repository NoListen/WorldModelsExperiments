import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFla, onehot_actions
from env import make_env
import os

def load_sample_data(fn ,seq_len, na):
    raw_data = np.load(fn)
    n = len(raw_data["action"])
    idx = np.random.randint(0, n-seq_len+1) # the final one won't be taken
    a = raw_data["action"][idx:idx+seq_len]
    obs = raw_data["obs"][idx:idx+seq_len]

    oh_a = onehot_actions(a, na)
    return oh_a, obs

class DataSet(object):
    def __init__(self, seq_len, na, data_dir):
        fns = os.listdir(data_dir)
        self.fns = np.array([fn for fn in fns if '.npz' in fn])
        self.seq_len = seq_len
        self.na = na
        self.n = len(self.fns)
        self.ids = np.arange(self.n)
        self.i = 0
        np.random.shuffle(self.ids)

    def random_batch(self, batch_size):
        a = []
        obs = []
        indices = self.ids[self.i:self.i + batch_size]
        nb = len(indices)
        sample_fns = self.fns[indices]

        for fn in sample_fns:
            ta, tobs = load_sample_data(fn, self.seq_len, self.na)
            a.append(ta)
            obs.append(tobs)

        # reset.
        self.i += batch_size
        if self.i >= self.n:
            np.random.shuffle(self.ids)
            self.i = 0

        if nb < batch_size:
            # sample the data
            ta, tobs = self.random_batch(batch_size-nb)
            a = np.concatenate([a, ta], axis=0)
            obs = np.concatenate([obs, tobs], axis=0)
        return a, obs

# TODO determine whether joint learning will be better.
def learn(sess, z_size, data_dir, num_steps, max_seq_len,
          batch_size=32, rnn_size=256,
          grad_clip=1.0, n_mix=3, vae_lr=0.0001,
          lr=0.001, min_lr=0.00001, decay=0.99,
          vae_path="tf_vae/final_vae.p",
          model_dir="tf_rnn", layer_norm=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):

    check_dir(model_dir)
    configure("%s/%s_j_rnn" % (model_dir, env_name))

    # define env
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    # build vae
    vae = ConvVAE(name="conv_vae",
                  z_size=z_size,
                  batch_size=1000)
    loadFromFlat(vae.get_variables(), vae_path)

    # build rnn
    rnn = MDNRNN("rnn",
                 num_steps,
                 max_seq_len+4, # 4 for the recent frames
                 input_size,
                 output_size,
                 batch_size,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 n_mix,  # number of mixtures in MDN
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp)

    # TODO create the dataset class.
    dataset = DataSet(max_seq_len+4, na, data_dir)

    # TODO option 1 iterative learning.
    # TODO sub define the loss.
    # Learn RNN to warm up.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    flat_target = tf.reshape(rnn.output_x[:, 4:, :], [-1, 1])

    loss = get_lossfunc(rnn.out_logmix[4:, :], rnn.out_mean[4:, :],
                        rnn.out_logstd[4:, :], flat_target)
    loss = tf.reduce_mean(loss)

    tf_lr = tf.Variable(lr, trainable=False)
    optimizer = tf.train.AdamOptimizer(tf_lr)

    gvs = optimizer.compute_gradients(loss)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    # train optimizer
    train_op = optimizer.apply_gradients(clip_gvs, global_step=global_step, name='train_step')

    sess.run(tf.global_variables_initializer())
    curr_lr = lr

    # TODO option 2 joint training.


    sess.run(tf.global_variables_initializer())
    loadFromFlat(vae.get_variables(), vae_path)

    # train loop:
    start = time.time()
    for local_step in range(num_steps):

        step = sess.run(global_step)
        curr_lr = (curr_lr - min_lr) * decay + min_lr

        raw_z, raw_a = dataset.random_batch(batch_size)
        inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
        outputs = raw_z[:, 1:, :]  # teacher forcing (shift by one predictions)

        feed = {rnn.input_x: inputs, rnn.output_x: outputs, tf_lr: curr_lr}
        (train_cost, state, train_step, _) = sess.run([loss, rnn.final_state, global_step, train_op], feed)
        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            log_value("training loss", train_cost, int(step // 20))
            output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (
                step, curr_lr, train_cost, time_taken)
            print(output_log)

    saveToFlat(rnn.get_variables(), model_dir + '/final_rnn.p')


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
    parser.add_argument('--data-dir', default="record", help="the directory of data")
    parser.add_argument('--max-seq-len', type=int, default=25, help="the maximum steps of dynamics to catch")
    parser.add_argument('--num-steps', default=4000, help="number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size")
    parser.add_argument('--rnn-size', type=int, default=256, help="rnn hidden state size")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="grad clip range")
    parser.add_argument('--n-mix', type=int, default=3, help="the number of gaussians in MDNRNN")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--min-lr', type=float, default=0.00001, help="minimum of learning rate")
    parser.add_argument('--decay', type=float, default=0.99999, help="decay of learning rate")

    # to load
    # Transfer the data directly
    parser.add_argument('--vae-lr', type=int,default=0.0001, help="the learning rate of vae")
    parser.add_argument('--vae-path', default="tf_vae/final_vae.p", help="the vae model to load")
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

