import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # can just override for multi-gpu systems

import time
import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from rnn.rnn import MDNRNN
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFlat,\
    onehot_actions, check_dir, get_lossfunc
from env import make_env
import os
from tensorboard_logger import configure, log_value
from config import env_name

class Clipper(object):
  def __init__(self, output_size, n_mix, seq_len, start):
    self.n_mix = n_mix
    self.p_size = n_mix * output_size
    self.seq_len = seq_len
    self.start = start

  def clip(self, o):
    o = tf.reshape(o, [-1, self.seq_len, self.p_size])
    o = tf.reshape(o[:, self.start:, :], [-1, self.n_mix])
    return o

class DataSet(object):
    def __init__(self, seq_len, na, data_dir):
        self.data_dir = data_dir
        fns = os.listdir(data_dir)
        self.fns = np.array([fn for fn in fns if '.npz' in fn])
        self.seq_len = seq_len
        self.na = na
        self.n = len(self.fns)
        self.ids = np.arange(self.n)
        self.i = 0
        np.random.shuffle(self.ids)

    def random_batch(self, batch_size):
        obs = []
        a = []
        indices = self.ids[self.i:self.i + batch_size]
        nb = len(indices)
        sample_fns = self.fns[indices]

        for fn in sample_fns:
            ta, tobs = self.load_sample_data(fn, self.seq_len, self.na)
            a.append(ta)
            obs.append(tobs)

        # reset.
        self.i += batch_size
        if self.i >= self.n:
            np.random.shuffle(self.ids)
            self.i = 0

        obs = np.array(obs)
        obs = np.expand_dims(obs, axis=-1)/255.
        a = np.array(a)
        if nb < batch_size:
            # sample the data
            tobs, ta = self.random_batch(batch_size-nb)
            a = np.concatenate([a, ta], axis=0)
            obs = np.concatenate([obs, tobs], axis=0)
        return obs, a

    def load_sample_data(self, fn, seq_len, na):
        raw_data = np.load(self.data_dir+'/'+fn)
        n = len(raw_data["action"])
        idx = np.random.randint(0, n - seq_len)  # the final one won't be taken
        a = raw_data["action"][idx:idx + seq_len+1] # sample one more.
        obs = raw_data["obs"][idx:idx + seq_len+1] # sample one more

        oh_a = onehot_actions(a, na)
        return oh_a, obs

# TODO determine whether joint learning will be better.
def learn(sess, z_size, data_dir, num_steps, max_seq_len,
          batch_size=32, rnn_size=256,
          grad_clip=1.0, n_mix=3, vae_lr=0.0001, kl_tolerance=0.5,
          lr=0.001, min_lr=0.00001, decay=0.99,
          vae_path="tf_vae/final_vae.p",
          model_dir="tf_rnn", layer_norm=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):

    # constant
    # tran_steps = 4

    check_dir(model_dir)
    configure("%s/%s_j_rnn" % (model_dir, env_name))

    # define env
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    # build dataset
    dataset = DataSet(max_seq_len+4, na, data_dir)

    # build vae
    vae = ConvVAE(name="conv_vae",
                  z_size=z_size,
                  batch_size=(dataset.seq_len+1)*batch_size) # 32 * 29
    # build the graph
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="x")
    z = vae.build_encoder(x)
    y = vae.build_decoder(z)

    tf_r_loss = -tf.reduce_sum(x * tf.log(y + 1e-8) +
                               (1. - x) * (tf.log(1. - y + 1e-8)), [1, 2, 3])
    tf_r_loss = tf.reduce_mean(tf_r_loss)
    tf_kl_loss = - 0.5 * tf.reduce_sum(
        (1 + vae.logvar - tf.square(vae.mu) - tf.exp(vae.logvar)),
        axis=1)
    tf_kl_loss = tf.reduce_mean(tf.maximum(tf_kl_loss, kl_tolerance * z_size))
    tf_vae_loss = tf_kl_loss + tf_r_loss

    vae_var_list = vae.get_variables()
    # no decay
    vae_opt = tf.train.AdamOptimizer(vae_lr)
    vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_var_list)  # can potentially clip gradients here.

    # option only update part of the gradients.
    # Then I might need to specify those grads.

    # the first optimizer
    vae_train_op = vae_opt.apply_gradients(
        vae_grads, name='train_step')
    # ---------------------------------------------------------------- #

    # build rnn

    # TODO option 1 iterative learning.
    # Learn RNN to warm up.

    # rnn loss
    # 1. the difference between z1 and z2
    # TODO 2. the difference between s1 and s2

    tf_a = tf.placeholder(tf.float32, shape=[None, dataset.seq_len+1, na])
    rnn_z = tf.placeholder(dtype=tf.float32, shape=[batch_size, dataset.seq_len+1, z_size])
    input_z = rnn_z[:, :-1, :]
    output_z = rnn_z[:, 1:, :]
    input_x = tf.concat([input_z, tf_a[:, :-1, :]], axis=2)
    output_x = output_z

    rnn = MDNRNN("rnn",
                 max_seq_len + 4,  # 4 for the recent frames
                 input_size,
                 output_size,
                 batch_size,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 n_mix,  # number of mixtures in MDN
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp)

    clipper = Clipper(output_size, n_mix, dataset.seq_len, 4)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # phase 1
    out_logmix, out_mean, out_logstd, initial_state, final_state = rnn.build_model(input_x)

    out_logmix = clipper.clip(out_logmix)
    out_mean = clipper.clip(out_mean)
    out_logstd = clipper.clip(out_logstd)


    flat_target = tf.reshape(output_x[:, 4:, :], [-1, 1])

    rnn_loss1 = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target)
    rnn_loss1 = tf.reduce_mean(rnn_loss1)


    rnn_var_list = rnn.get_variables()
    tf_lr = tf.Variable(lr, trainable=False)
    rnn_opt1 = tf.train.AdamOptimizer(tf_lr, name="warmup")

    gvs = rnn_opt1.compute_gradients(rnn_loss1, rnn_var_list)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    # train optimizer
    rnn_train_op1 = rnn_opt1.apply_gradients(clip_gvs, global_step=global_step, name='rnn_op2')

    # phase 2
    rnn_z_with_vae = tf.reshape(z, (batch_size, dataset.seq_len+1, z_size))
    # overwrite
    input_z = rnn_z_with_vae[:, :-1, :]
    output_z = tf.stop_gradient(rnn_z_with_vae[:, 1:, :])
    input_x = tf.concat([input_z, tf_a[:, :-1, :]], axis=2)
    output_x = output_z
    flat_target = tf.reshape(output_x[:, 4:, :], [-1, 1])

    out_logmix_with_vae, out_mean_with_vae, out_logstd_with_vae, \
        initial_state_with_vae, final_state_with_vae = rnn.build_model(input_x, reuse=True)

    out_logmix_with_vae = clipper.clip(out_logmix_with_vae)
    out_mean_with_vae = clipper.clip(out_mean_with_vae)
    out_logstd_with_vae = clipper.clip(out_logstd_with_vae)

    rnn_loss2 = get_lossfunc(out_logmix_with_vae, out_mean_with_vae,
                             out_logstd_with_vae, flat_target)
    rnn_opt2 = tf.train.AdamOptimizer(tf_lr, name="joint")
    vae_opt2 = tf.train.AdamOptimizer(vae_lr, name="vae_joint")
    # overwrite
    gvs = rnn_opt2.compute_gradients(rnn_loss2, rnn_var_list)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs if grad is not None]
    # train optimizer
    rnn_train_op2 = rnn_opt2.apply_gradients(clip_gvs, global_step=global_step, name='rnn_op2')

    gvs = vae_opt2.compute_gradients(rnn_loss2, vae_var_list)
    vae_train_op2 = vae_opt2.apply_gradients(gvs, name="vae_op2")

    sess.run(tf.global_variables_initializer())
    curr_lr = lr


    # TODO suboption 1 warmup and joint learn.


    # initialize and load the model
    sess.run(tf.global_variables_initializer())
    loadFromFlat(vae_var_list, vae_path)

    # TODO  the learning process is divided into two parts
    # 1. learn from series -> normally ( randomly clip the sequence )
    # 2. learn from record -> generate the data in time.


    warmup_num_steps = num_steps//2
    joint_num_steps = num_steps - warmup_num_steps


    # warmup
    start = time.time()
    for i in range(warmup_num_steps):

        step = sess.run(global_step)
        curr_lr = (curr_lr - min_lr) * decay + min_lr

        raw_obs, raw_a = dataset.random_batch(batch_size)
        raw_obs = raw_obs.reshape((-1,) + raw_obs.shape[2:])
        # the grads won't be back propagated
        raw_z = sess.run(z, feed_dict={x: raw_obs})
        raw_z = raw_z.reshape((batch_size, dataset.seq_len+1, z_size))

        feed = {rnn_z: raw_z, tf_a: raw_a, tf_lr: curr_lr}
        (train_cost, train_step, _) = sess.run([rnn_loss1, global_step, rnn_train_op1], feed)
        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            log_value("training loss", train_cost, int(step // 20))
            output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (
                step, curr_lr, train_cost, time_taken)
            print(output_log)

    print("Begin Joint Training..")

    # TODO try option 2
    # joint training
    for i in range(joint_num_steps):

        step = sess.run(global_step)
        curr_lr = (curr_lr - min_lr) * decay + min_lr

        raw_obs, raw_a = dataset.random_batch(batch_size)
        raw_obs = raw_obs.reshape((-1,) + raw_obs.shape[2:])
        # the grads won't be back propagated

        feed = {x: raw_obs, tf_a: raw_a, tf_lr: curr_lr}
        # alternative training
        # (train_cost, train_step, _) = sess.run([rnn_loss2, global_step, rnn_train_op2], feed)
        # vae_train_cost, _ = sess.run([tf_vae_loss, vae_train_op], feed={x: raw_obs})

        # joint training
        (train_cost, train_step, _, vae_train_cost, _, _) = sess.run([rnn_loss2, global_step, rnn_train_op2,
                                                                   tf_vae_loss, vae_train_op, vae_train_op2], feed)

        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            log_value("training loss", train_cost, int(step // 20))
            output_log = "step: %d, lr: %.6f, cost: %.4f, vae_cost: %.4f, train_time_taken: %.4f" % (
                step, curr_lr, train_cost, vae_train_cost, time_taken)
            print(output_log)

    saveToFlat(rnn_var_list, model_dir + '/final_rnn.p')


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
    parser.add_argument('--kl-tolerance', type=float, default=0.5, help="kl tolerance")

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

