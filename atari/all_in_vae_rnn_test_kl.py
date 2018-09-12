import os
from collections import namedtuple
#os.environ["CUDA_VISIBLE_DEVICES"]="2" # can just override for multi-gpu systems
import pickle
import time
import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from rnn.vrnn import VRNN
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFlat,\
    onehot_actions, check_dir, get_lr_lossfunc,  get_kl_lossfunc, get_kl2normal_lossfunc
from env import make_env
import os
from tensorboard_logger import configure, log_value
from config import env_name
import copy
import random
from collections import defaultdict
from wrappers import DatasetTransposeWrapper, DatasetSwapWrapper, DatasetHorizontalConcatWrapper, DatasetVerticalConcatWrapper
VAE_COMP = namedtuple('VAE_COMP', ['a', 'x', 'y', 'z', 'mu', 'logstd', 'ma', 'mx', 'my', 'mz', 'mmu', 'mlogstd', 
                                    'r_loss', 'kl_loss', 'loss', 'var_list', 'fc_var_list', 'train_op'])
RNN_COMP_WITH_OPT = namedtuple('RNN_COMP', ['z_input', 'a', 'logmix', 'mean', 'logstd', 'var_list'])
RNN_COMP_WITH_VAE = namedtuple("RNN_COMP_WITH_VAE", ['logstd', 'mean', 'loss', 'pz', 'kl2vae'])

def get_kl(mean, logstd, target_mean, target_logstd):
    return np.mean(logstd - target_logstd + (np.exp(2*target_logstd)+np.square(target_mean-mean))/2/np.exp(2*logstd) - 0.5, axis=0)

class DataSet(object):
    def __init__(self, seq_len, na, data_dir, fns):
        self.data_dir = data_dir
        self.fns = np.array(fns)
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
            tobs, ta = self.load_sample_data(fn, self.seq_len, self.na)
            obs.append(tobs)
            a.append(ta)

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
        return obs, oh_a

class DatasetManager(object):
    def __init__(self, datasets):
        self.datasets = datasets

    def random_batch(self, batch_size_per_task):
        obs_list, a_list = [], []
        for d in self.datasets:
            tobs, ta = d.random_batch(batch_size_per_task)
            obs_list.append(tobs)
            a_list.append(ta)
        return obs_list, a_list

# TODO Option 1 update VAE in meta-learning phase.
# (Processing...) Option 2 update VAE seperately. (This can also work but without update RNN.)
def build_vae(name, vae, na, z_size, seq_len, vae_lr, kl_tolerance, fc_limit=False):
    # used for later input tnesor
    a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name+"_a")
    ma = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name+"_ma")

    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name=name+"_x")
    mx = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name=name+"_mx")

    mu, logstd, z = vae.build_encoder(x, reuse=False)
    y = vae.build_decoder(z, reuse=False)

    # in meta learning, we also received gradients from RNN.
    mmu, mlogstd, mz = vae.build_encoder(mx, reuse=True)
    my = vae.build_decoder(mz, reuse=True)

    tf_r_loss = -tf.reduce_sum(x * tf.log(y + 1e-8) +
                               (1. - x) * (tf.log(1. - y + 1e-8)), [1, 2, 3])
    tf_r_loss = tf.reduce_mean(tf_r_loss)
    tf_kl_loss = - 0.5 * tf.reduce_sum((1 + logstd - tf.square(mu)
                                        - tf.exp(logstd)), axis=1)
    tf_kl_loss = tf.reduce_mean(tf.maximum(tf_kl_loss, kl_tolerance * z_size))
    tf_vae_loss = tf_kl_loss + tf_r_loss
    vae_var_list = vae.get_variables()
    vae_fc_var_list = vae.get_fc_variables()

    # no decay
    vae_opt = tf.train.AdamOptimizer(vae_lr)
    if fc_limit:
      vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_fc_var_list)
    else:
      vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_var_list)
    vae_train_op = vae_opt.apply_gradients(vae_grads, name=name+'train_op')

    vae_comp = VAE_COMP(a=a, x=x, z=z, y=y, mu=mu, logstd=logstd, ma=ma, mx=mx, mz=mz, my=my,
                        mmu=mmu, mlogstd=mlogstd, r_loss=tf_r_loss, kl_loss=tf_kl_loss, loss=tf_vae_loss,
                        var_list=vae_var_list, fc_var_list=vae_fc_var_list,
                        train_op=vae_train_op)
    return vae_comp

# TODO determine whether joint learning will be better.
def learn(sess, n_tasks, z_size, data_dir, num_steps, max_seq_len,
          batch_size_per_task=16, rnn_size=256,
          grad_clip=1.0, v_lr=0.0001, vr_lr=0.0001,
          min_v_lr=0.00001, v_decay=0.999, kl_tolerance=0.5,
          lr=0.001, min_lr=0.00001, decay=0.999,
          transform="transpose", vae_dir="tf_vae",
          model_dir="tf_rnn", layer_norm=False,
          fc_limit=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):
    # TODO remove this limit.
    batch_size = batch_size_per_task * n_tasks


    if transform == "transpose":
      wrapper = DatasetTransposeWrapper
    elif transform == "swap":
      wrapper = DatasetSwapWrapper
    elif transform == "concat1":
      wrapper = DatasetHorizontalConcatWrapper
    elif transform == "concat2":
      wrapper = DatasetVerticalConcatWrapper
    else:
      raise Exception("Such transform is not available")

    print("Batch size for each taks is", batch_size_per_task)
    print("The total batch size is", batch_size)

    check_dir(model_dir)
    configure("%s/%s_rnn" % (model_dir, env_name))

    # define env
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    print("the environment", env_name, "has %i actions" % na)

    # build dataset
    fns = os.listdir(data_dir)
    fns = [fn for fn in fns if '.npz' in fn]

    dataset = DataSet(max_seq_len+4, na, data_dir, fns)
    seq_len = dataset.seq_len

    print("The datasets has been created")

   
    tf_v_lr = tf.placeholder(tf.float32, shape=[]) # learn from reconstruction.
    tf_vr_lr = tf.placeholder(tf.float32, shape=[]) # learn from vr

    vaes = []
    vae_comps = []
    for i in range(n_tasks):
        vae = ConvVAE(name="vae%i" % i,
                      z_size=z_size,
                      batch_size=(seq_len + 1) * batch_size_per_task)
        vae_comp = build_vae("vae%i" % i, vae, na, z_size, seq_len, tf_v_lr, kl_tolerance, fc_limit)
        vaes.append(vae)
        vae_comps.append(vae_comp)


    comp = vae_comps[1]
    ty = vaes[0].build_decoder(comp.z, reuse=True)
    tty = wrapper.transform(ty)
    transform_loss = -tf.reduce_sum(comp.x * tf.log(tty + 1e-8) +
                               (1. - comp.x) * (tf.log(1. - tty + 1e-8)), [1, 2, 3])
    # TODO add one in the RNN's prediction error.
    transform_loss = tf.reduce_mean(transform_loss)
    vae_total_loss = tf.reduce_mean([comp.loss for comp in vae_comps])

    sess.run(tf.global_variables_initializer())

    # initialize and load the model
    sess.run(tf.global_variables_initializer())
    joint_num_steps = num_steps

    print("Begin Pretraining..")
    


    # todo 1 rank all the directories.x`
    dns = os.listdir(model_dir)
    dns = [dn for dn in dns if 'it' in dn]
    # I want to store all data in one dictionary and store it using pickle
    # start from one.
    # log scale.

    ids = [int(dn[3:]) for dn in dns]
    max_id = np.max(ids)
    #max_ids = np.max(ids)

    #for i in range(1):
    #for i in range(0, max_id//10 + 1):
    for i in range(37, 38):
      dn = model_dir + '/it_' + str(i*10)
      # Load the model 
      for j, comp in enumerate(vae_comps):
        loadFromFlat(comp.var_list, dn+"/vae%i.p" % j)

      vae_costs = []
      transform_costs = []
      kls = []
      #for _ in range(10):
      for _ in range(1):
          raw_obs, raw_a = dataset.random_batch(batch_size_per_task)
          raw_obs = raw_obs.reshape((-1,) + raw_obs.shape[2:])

          feed = {}
          feed[vae_comps[0].x] =  raw_obs
          feed[vae_comps[1].x] =  wrapper.data_transform(raw_obs)

          (vae_cost, transform_cost, z, z1, logstd, logstd1, mu, mu1) = sess.run([vae_total_loss, transform_loss,
                                                vae_comps[0].z, vae_comps[1].z,
                                                vae_comps[0].logstd, vae_comps[1].logstd,
                                                vae_comps[0].mu, vae_comps[1].mu], feed)
          vae_costs.append(vae_cost)
          transform_costs.append(transform_cost)
          kls.append(get_kl(mu1, logstd1, mu, logstd))

      output_log = "model: %i, vae cost: %.2f, transform cost: %.2f, z distance: %.2f" % \
                         (i, np.mean(vae_costs), np.mean(transform_costs), np.mean(np.sum(np.abs(z-z1), axis=1)))
      print(output_log)
      print(np.mean(kls, axis=0))
      print(np.mean(np.abs(mu), axis=0))
      print(np.mean(np.abs(mu1), axis=0))
      print(np.mean(np.abs(logstd), axis=0))
      print(np.mean(np.abs(logstd1), axis=0))
      print(np.mean(z, axis=0))
      print(np.mean(z1, axis=0))
      print(np.mean(np.abs(z-z1), axis=0))

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
    parser.add_argument('--data-dir', default="record", help="the directory of data")
    parser.add_argument('--max-seq-len', type=int, default=25, help="the maximum steps of dynamics to catch")
    parser.add_argument('--num-steps', type=int, default=4000, help="number of training iterations")

    parser.add_argument('--batch-size-per-task', type=int, default=16, help="batch size for each task")

    parser.add_argument('--rnn-size', type=int, default=256, help="rnn hidden state size")
    parser.add_argument('--grad-clip', type=float, default=1.0, help="grad clip range")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--min-lr', type=float, default=0.00001, help="minimum of learning rate")
    parser.add_argument('--decay', type=float, default=0.999, help="decay of learning rate")

    parser.add_argument('--transform', default="transpose", help="type of transform. ['transform', 'swap', 'concat1', 'concat2']")

    # to load
    # Transfer the data directly
    parser.add_argument('--n-tasks', type=int, default=2, help="the number of tasks")
    # parser.add_argument('--n-updates', type=int, default=1, help="number of inner gradient updates during training")
    parser.add_argument('--v-lr', type=float, default=0.0001, help="the learning rate of vae")
    parser.add_argument('--vr-lr', type=float, default=0.0001, help="the learning rate of vae to reduce the rnn loss")
    parser.add_argument('--min-v-lr', type=float, default=0.00001, help="the minimum of vae learning rate")
    parser.add_argument('--v-decay', type=float, default=0.999, help="the decay of vae learning rare")

    parser.add_argument('--vae-dir', default="tf_vae", help="the path of vae models to load")
    parser.add_argument('--kl-tolerance', type=float, default=0.5, help="kl tolerance")

    parser.add_argument('--model-dir', default="tf_rnn", help="the directory to store rnn model")
    parser.add_argument('--layer-norm', action="store_true", default=False, help="layer norm in RNN")
    parser.add_argument('--fc-limit', action="store_true", default=False, help="limit training the fc layers in vae")
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

