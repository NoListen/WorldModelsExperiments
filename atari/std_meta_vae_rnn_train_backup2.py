import os
from collections import namedtuple
#os.environ["CUDA_VISIBLE_DEVICES"]="2" # can just override for multi-gpu systems

import time
import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from rnn.vrnn import VRNN
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFlat,\
    onehot_actions, check_dir, get_lr_lossfunc, get_kl2normal_lossfunc
from env import make_env
import os
from tensorboard_logger import configure, log_value
from config import env_name
import copy
import random

VAE_COMP = namedtuple('VAE_COMP', ['a', 'x', 'y', 'z', 'mu', 'logstd', 'ma', 'mx', 'my', 'mz', 'mmu', 'mlogstd', 
                                    'r_loss', 'kl_loss', 'loss', 'var_list', 'fc_var_list', 'train_op'])
RNN_COMP_WITH_OPT = namedtuple('RNN_COMP', ['z_input', 'a', 'logmix', 'mean', 'logstd', 'var_list'])

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

class DatasetTransposeWrapper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def random_batch(self, batch_size):
        obs, a = self.dataset.random_batch(batch_size)
        obs = np.transpose(obs, [0, 1, 3, 2, 4])
        return obs, a

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
def build_vae(name, vae, na, z_size, seq_len, vae_lr, kl_tolerance):
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
    # vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_var_list)
    vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_fc_var_list)
    vae_train_op = vae_opt.apply_gradients(vae_grads, name=name+'train_op')

    vae_comp = VAE_COMP(a=a, x=x, z=z, y=y, mu=mu, logstd=logstd, ma=ma, mx=mx, mz=mz, my=my,
                        mmu=mmu, mlogstd=mlogstd, r_loss=tf_r_loss, kl_loss=tf_kl_loss, loss=tf_vae_loss,
                        var_list=vae_var_list, fc_var_list=vae_fc_var_list,
                        train_op=vae_train_op)
    return vae_comp

# Just build the structure.
def build_rnn(name, rnn, na, z_size, batch_size, seq_len):
    a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name+"_a")
    rnn_z = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len, z_size], name=name+"_z")


    input_x = tf.concat([rnn_z, a], axis=2)
    out_logmix, out_mean, out_logstd = rnn.build_model(input_x)

    rnn_var_list = rnn.get_variables()
    rnn_comp = RNN_COMP_WITH_OPT(a=a, z_input=rnn_z, logmix=out_logmix,
                                 mean=out_mean, logstd=out_logstd, 
                                var_list=rnn_var_list)

    return rnn_comp


# Do we need another placeholder ?

def process_z_with_vae(x, z, a, batch_size, seq_len, z_size):
    # reshape and cut
    target_y = tf.reshape(x, (batch_size, seq_len+1, 64, 64, 1))[:, 5:, ...]
    target_y = tf.reshape(target_y, (-1, 64, 64, 1))

    input_z = tf.reshape(z, (batch_size, seq_len+1, z_size))[:, :-1, :]
    input_z = tf.concat([input_z, a], axis=2)

    return input_z, target_y

def rnn_with_vae(vae, rnn, x, z, a, rnn_lv_dict, z_size, batch_size, seq_len, kl_tolerance):
    input_z, target_y = process_z_with_vae(x, z, a, batch_size, seq_len, z_size)
    
    pz, mean, logstd = rnn.build_variant_model(input_z, rnn_lv_dict, reuse=True)
    mean = tf.reshape(mean, [-1, z_size])
    logstd = tf.reshape(logstd, [-1, z_size])
    pz = tf.reshape(pz, [batch_size, seq_len, z_size])[:, 4:, :]
    pz = tf.reshape(pz, [-1, z_size])
    py = vae.build_decoder(pz, reuse=True) # -1, 64, 64, 1
    rnn_loss = tf.reduce_mean(get_lr_lossfunc(target_y, py))
    rnn_kl_loss = get_kl2normal_lossfunc(mean, logstd)
    rnn_loss += tf.reduce_mean(tf.maximum(rnn_kl_loss, kl_tolerance * z_size))
    return rnn_loss, logstd

# Meta part.
def build_rnn_with_vae(vae, rnn, rnn_lv_dict, comp, z_size, seq_len, batch_size, rnn_lr, kl_tolerance=0.5):
    rnn_loss, logstd = rnn_with_vae(vae, rnn, comp.x, comp.z, comp.a, rnn_lv_dict,
                                       z_size, batch_size, seq_len, kl_tolerance)

    grads = tf.gradients(rnn_loss, list(rnn_lv_dict.values()))
    grads = dict(zip(rnn_lv_dict.keys(), grads))
    for k in rnn_lv_dict.keys():
        rnn_lv_dict[k] = rnn_lv_dict[k] - rnn_lr * grads[k]

    rnn_meta_loss, meta_logstd = rnn_with_vae(vae, rnn, comp.mx, comp.mz, comp.ma, rnn_lv_dict,
                                       z_size, batch_size, seq_len, kl_tolerance)

    return rnn_loss, rnn_meta_loss, logstd, meta_logstd

# TODO determine whether joint learning will be better.
def learn(sess, n_tasks, z_size, data_dir, num_steps, max_seq_len,
          batch_size_per_task=16, rnn_size=256,
          grad_clip=1.0, vae_lr=0.0001, kl_tolerance=0.5,
          lr=0.001, min_lr=0.00001, decay=0.99,
          vae_dir="tf_vae",
          model_dir="tf_rnn", layer_norm=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):
    # TODO remove this limit.
    batch_size = batch_size_per_task * n_tasks

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
    random.shuffle(fns)
    fns1 = fns[:len(fns)//2]
    fns2 = fns[len(fns)//2:]

    dataset1 = DataSet(max_seq_len+4, na, data_dir, fns1)
    dataset2 = DataSet(max_seq_len+4, na, data_dir, fns2)

    datasets = [dataset1, DatasetTransposeWrapper(dataset2)]
    dm = DatasetManager(datasets) # sample from this one.
    seq_len = dataset1.seq_len

    print("The datasets has been created")


    vaes = []
    vae_comps = []
    for i in range(n_tasks):
        vae = ConvVAE(name="vae%i" % i,
                      z_size=z_size,
                      batch_size=(seq_len + 1) * batch_size_per_task)
        vae_comp = build_vae("vae%i" % i, vae, na, z_size, seq_len, vae_lr, kl_tolerance)
        vaes.append(vae)
        vae_comps.append(vae_comp)


    comp = vae_comps[0]
    ty = vaes[1].build_decoder(comp.z, reuse=True)
    tty = tf.transpose(ty, [0, 2, 1, 3])
    transpose_loss = -tf.reduce_sum(comp.x * tf.log(tty + 1e-8) +
                               (1. - comp.x) * (tf.log(1. - tty + 1e-8)), [1, 2, 3])
    transpose_loss = tf.reduce_mean(transpose_loss)
    vae_all_op = tf.group([comp.train_op for comp in vae_comps])
    vae_total_loss = tf.reduce_sum([comp.loss for comp in vae_comps])

    print("The all vaes have been created")

    # Meta RNN.
    rnn = VRNN("rnn",
                 max_seq_len + 4,  # 4 for the recent frames
                 input_size,
                 output_size,
                 batch_size_per_task,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp)


    global_step = tf.Variable(0, name='global_step', trainable=False)

    tf_lr = tf.placeholder(tf.float32, shape=[])
    # Just build the architecture.
    rnn_comp = build_rnn("rnn", rnn, na, z_size, batch_size_per_task, seq_len)
    rnn_lv_dict = rnn.get_linear_variables()

    print("The basic rnn has been built")

    # phase 2
    rnn_losses = []
    rnn_meta_losses = []
    vae_meta_var_list = [] # VAE receives the gradients from meta loss
    rnn_logstds = []
    rnn_meta_logstds = []
    for i in range(n_tasks):
        comp = vae_comps[i]
        vae_meta_var_list += comp.fc_var_list
        vae = vaes[i]

        tmp_rnn_lv_dict = copy.copy(rnn_lv_dict)
        rnn_loss, rnn_meta_loss, rnn_logstd, rnn_meta_logstd = build_rnn_with_vae(vae, rnn, tmp_rnn_lv_dict, comp, z_size,
                                               seq_len, batch_size_per_task, tf_lr)

        rnn_logstds.append(rnn_logstd)
        rnn_meta_logstds.append(rnn_meta_logstd)
        rnn_losses.append(rnn_loss)
        rnn_meta_losses.append(rnn_meta_loss)

    print("RNN has been connected to each VAE")
    verified_rnn_var_list=  rnn.get_variables()
    rnn_all_vn_list = [v.name for v in verified_rnn_var_list]
    rnn_vn_list = [v.name for v in verified_rnn_var_list if 'Adam' not in v.name]
    print("The all variables in rnn now are", rnn_all_vn_list)
    print("The meta variables in rnn now are", rnn_vn_list)

    # We are going to minimize this
    # TODO the gradients part include the VAE. (It should be computed in adapted version.)
    rnn_meta_total_loss = tf.reduce_mean(rnn_meta_losses)
    rnn_total_loss = tf.reduce_mean(rnn_losses)

    rnn_mean_logstd = tf.reduce_mean(rnn_logstds)
    rnn_mean_meta_logstd = tf.reduce_mean(rnn_meta_logstds)
    vae_mean_logstd = tf.reduce_mean([comp.logstd for comp in vae_comps])


    rnn_wu_opt = tf.train.AdamOptimizer(tf_lr, name="wu_rnn_opt")
    gvs = rnn_wu_opt.compute_gradients(rnn_total_loss, rnn_comp.var_list)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs if grad is not None]
    rnn_wu_op = rnn_wu_opt.apply_gradients(clip_gvs, global_step=global_step, name='rnn_wu_op')
    

    rnn_meta_opt = tf.train.AdamOptimizer(tf_lr, name="meta_rnn_opt")
    gvs = rnn_meta_opt.compute_gradients(rnn_meta_total_loss, rnn_comp.var_list)
    clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs if grad is not None]
    # train optimizer
    rnn_meta_op = rnn_meta_opt.apply_gradients(clip_gvs, global_step=global_step, name='rnn_meta_op')

    vae_meta_opt = tf.train.AdamOptimizer(vae_lr/4, name="meta_vae_opt")
    #gvs = vae_meta_opt.compute_gradients(rnn_meta_total_loss, vae_meta_var_list)
    gvs = vae_meta_opt.compute_gradients(rnn_total_loss, vae_meta_var_list)
    vae_meta_op = vae_meta_opt.apply_gradients(gvs, name='vae_meta_op')

    sess.run(tf.global_variables_initializer())
    curr_lr = lr

    # initialize and load the model
    sess.run(tf.global_variables_initializer())
    for i, comp in enumerate(vae_comps):
        loadFromFlat(comp.var_list, vae_dir+"/vae%i.p" % i)

    if os.path.exists(model_dir+'/base_rnn.p'):
        loadFromFlat(rnn_comp.var_list, model_dir+'/base_rnn.p')
        warmup_num_steps = 0
    else:
        warmup_num_steps = num_steps//2
    joint_num_steps = num_steps - warmup_num_steps

    print("Begin Pretraining..")


    # TODO make sure pretraining has no problems
    start = time.time()
    for i in range(warmup_num_steps):

        step = sess.run(global_step)
        curr_lr = (curr_lr - min_lr) * decay + min_lr

        raw_obs_list, raw_a_list = dm.random_batch(batch_size_per_task)
        raw_obs_list = [obs.reshape((-1,) + obs.shape[2:]) for obs in raw_obs_list]
        # the grads won't be back propagated
        feed = {tf_lr: curr_lr}
        for j in range(n_tasks):
            comp = vae_comps[j]
            feed[comp.x] =  raw_obs_list[j]
            feed[comp.a] = raw_a_list[j][:, :-1, :]

        (rnn_cost, vae_cost, transpose_cost, rnn_logstd, vae_logstd, _) = sess.run([rnn_total_loss, vae_total_loss, transpose_loss,
                                                              rnn_mean_logstd, vae_mean_logstd , rnn_wu_op], feed)
        #(rnn_cost, vae_cost, rnn_logstd) = sess.run([rnn_total_loss, vae_total_loss, rnn_mean_logstd], feed)
        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            #log_value("training loss", train_cost, int(step // 20))
            output_log = "step: %d, lr: %.6f, rlstd:%.6f, vlstd:%.6f, rnn_cost: %.4f, vae_cost:%.4f, transpose_cost:%.4f" % \
                     (step, curr_lr, rnn_logstd, vae_logstd, rnn_cost, vae_cost, transpose_cost)
            print(output_log)

    if not  os.path.exists(model_dir+'/base_rnn.p'):
        saveToFlat(rnn_comp.var_list, model_dir+'/base_rnn.p')

    print("Begin Meta Training..")

    for i in range(joint_num_steps):

        step = sess.run(global_step)
        curr_lr = (curr_lr - min_lr) * decay + min_lr

        raw_obs_list, raw_a_list = dm.random_batch(batch_size_per_task)
        raw_obs_list = [obs.reshape((-1,) + obs.shape[2:]) for obs in raw_obs_list]

        meta_obs_list, meta_a_list = dm.random_batch(batch_size_per_task)
        meta_obs_list = [obs.reshape((-1,) + obs.shape[2:]) for obs in meta_obs_list]

        feed = {tf_lr: curr_lr}
        for j in range(n_tasks):
            comp = vae_comps[j]
            feed[comp.x] =  raw_obs_list[j]
            #feed[comp.mx] = meta_obs_list[j]
            feed[comp.a] = raw_a_list[j][:, :-1, :]
            #feed[comp.ma] = meta_a_list[j][:, :-1, :]

        # joint training
        """(meta_cost, rnn_cost, vae_cost, transpose_cost, rnn_logstd, rnn_meta_logstd, vae_logstd, _ , _, _) = sess.run([rnn_meta_total_loss,
                                                             rnn_total_loss, vae_total_loss, transpose_loss,
                                                              rnn_mean_logstd, rnn_mean_meta_logstd, vae_mean_logstd ,vae_all_op, rnn_meta_op,
                                                              vae_meta_op], feed)
        """
        (rnn_cost, vae_cost, transpose_cost, rnn_logstd, vae_logstd, _ , _, _) = sess.run([
                                                             rnn_total_loss, vae_total_loss, transpose_loss,
                                                              rnn_mean_logstd,  vae_mean_logstd ,vae_all_op, 
                                                              rnn_wu_op, vae_meta_op], feed)
        if (step % 20 == 0 and step > 0):
            end = time.time()
            time_taken = end - start
            start = time.time()
            #log_value("training loss", meta_cost, int(step // 20))
            #output_log = "step: %d, lr: %.6f, meta cost: %.2f, vae cost: %.2f, " \
            #             "rnn cost: %.2f, transpose cost: %.2f, rstd:%.4f, mrstd:%.4f, vstd:%.4f" % \
            #             (step, curr_lr, meta_cost, vae_cost, rnn_cost, transpose_cost, rnn_logstd, rnn_meta_logstd, vae_logstd)
            output_log = "step: %d, lr: %.6f, vae cost: %.2f, " \
                         "rnn cost: %.2f, transpose cost: %.2f, rstd:%.4f, vstd:%.4f" % \
                         (step, curr_lr, vae_cost, rnn_cost, transpose_cost, rnn_logstd, vae_logstd)
            print(output_log)

    saveToFlat(rnn_comp.var_list, model_dir + '/final_rnn.p')
    for i in range(n_tasks):
      comp = vae_comps[i]
      saveToFlat(comp.var_list, model_dir + '/final_vae%i.p' % i) 

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
    parser.add_argument('--decay', type=float, default=0.99999, help="decay of learning rate")

    # to load
    # Transfer the data directly
    parser.add_argument('--n-tasks', type=int, default=2, help="the number of tasks")
    # parser.add_argument('--n-updates', type=int, default=1, help="number of inner gradient updates during training")
    parser.add_argument('--vae-lr', type=float, default=0.0001, help="the learning rate of vae")
    parser.add_argument('--vae-dir', default="tf_vae", help="the path of vae models to load")
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

