import os
from collections import namedtuple
#os.environ["CUDA_VISIBLE_DEVICES"]="2" # can just override for multi-gpu systems
import json
import time
import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vae.vae import ConvVAE
from rnn.vrnn import VRNN
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFlat, pad_num, \
    onehot_actions, check_dir, get_lr_lossfunc,  get_kl_lossfunc, get_kl2normal_lossfunc
from env import make_env
import os
from config import env_name
import copy
import random
from datetime import datetime
from wrappers import DatasetTransposeWrapper, DatasetSwapWrapper, DatasetHorizontalConcatWrapper,\
                       DatasetVerticalConcatWrapper, DatasetColorWrapper
from scipy.misc import imsave, imread

VAE_COMP = namedtuple('VAE_COMP', ['a', 'x', 'y', 'z', 'mu', 'logstd', 'ma', 'mx', 'my', 'mz', 'mmu', 'mlogstd',
                                    'r_loss', 'kl_loss', 'loss', 'var_list', 'fc_var_list', 'train_opt'])
RNN_COMP_WITH_OPT = namedtuple('RNN_COMP', ['z_input', 'a', 'logmix', 'mean', 'logstd', 'var_list'])
RNN_COMP_WITH_VAE = namedtuple("RNN_COMP_WITH_VAE", ['logstd', 'mean', 'loss', 'py',
                                                     'pz', 'kl2vae', 'state_in', 'last_state'])

np.random.seed(1234567)

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
    #if fc_limit:
    #  vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_fc_var_list)
    #else:
    #  vae_grads = vae_opt.compute_gradients(tf_vae_loss, vae_var_list)
    #vae_train_op = vae_opt.apply_gradients(vae_grads, name=name+'train_op')

    vae_comp = VAE_COMP(a=a, x=x, z=z, y=y, mu=mu, logstd=logstd, ma=ma, mx=mx, mz=mz, my=my,
                        mmu=mmu, mlogstd=mlogstd, r_loss=tf_r_loss, kl_loss=tf_kl_loss, loss=tf_vae_loss,
                        var_list=vae_var_list, fc_var_list=vae_fc_var_list,
                        train_opt=vae_opt)
    return vae_comp

# Just build the structure.
def build_rnn(name, rnn, na, z_size, batch_size, seq_len):
    a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name+"_a")
    rnn_z = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len, z_size], name=name+"_z")


    input_x = tf.concat([rnn_z, a], axis=2)
    print(input_x.shape)

    # useless later on.
    out_logmix, out_mean, out_logstd = rnn.build_model(input_x)

    rnn_var_list = rnn.get_variables()
    rnn_comp = RNN_COMP_WITH_OPT(a=a, z_input=rnn_z, logmix=out_logmix,
                                 mean=out_mean, logstd=out_logstd, 
                                var_list=rnn_var_list)

    return rnn_comp


# Do we need another placeholder ?

def process_z_with_vae(x, z, a, batch_size, seq_len, z_size):
    # reshape and cut
    target_y = tf.reshape(x, (batch_size, seq_len+1, 64, 64, 1))[:, 1:, ...]
    target_y = tf.reshape(target_y, (-1, 64, 64, 1))

    input_z = tf.reshape(z, (batch_size, seq_len+1, z_size))[:, :-1, :]
    input_z = tf.concat([input_z, a], axis=2)

    return input_z, target_y

def rnn_with_vae(vae, rnn, x, z, a, rnn_lv_dict, z_size, batch_size, seq_len, kl_tolerance):
    input_z, target_y = process_z_with_vae(x, z, a, batch_size, seq_len, z_size)

    c_in = tf.placeholder(tf.float32, (batch_size, rnn.rnn_size))
    h_in = tf.placeholder(tf.float32, (batch_size, rnn.rnn_size))
    state_in = [c_in, h_in]
    pz, mean, logstd, last_state = rnn.build_cont_model(input_z, c_in, h_in, reuse=True)
    
    mean = tf.reshape(mean, [-1, z_size])
    logstd = tf.reshape(logstd, [-1, z_size])
    pz = tf.reshape(pz, [-1, z_size])
    py = vae.build_decoder(pz, reuse=True) # -1, 64, 64, 1
    rnn_loss = tf.reduce_mean(get_lr_lossfunc(target_y, py))
    rnn_kl_loss = get_kl2normal_lossfunc(mean, logstd)
    rnn_loss += tf.reduce_mean(tf.maximum(rnn_kl_loss, kl_tolerance * z_size))
    return rnn_loss, mean, logstd, py, pz, state_in, last_state

def cut(x, batch_size, seq_len, z_size):
    x = tf.reshape(x, [batch_size, seq_len+1, z_size])[:, 1:, :]
    x = tf.reshape(x, [-1, z_size])
    return x


# Complete the RNN comp
def build_rnn_with_vae(vae, rnn, rnn_lv_dict, comp, z_size, seq_len, batch_size, rnn_lr, kl_tolerance=0.5):
    rnn_loss, mean, logstd, py, pz, state_in, last_state = rnn_with_vae(vae, rnn, comp.x, comp.z, comp.a,
                                                    rnn_lv_dict, z_size, batch_size, seq_len, kl_tolerance)

    target_mu = cut(comp.mu, batch_size, seq_len, z_size)
    target_logstd = cut(comp.logstd, batch_size, seq_len, z_size)
    kl2vae = tf.reduce_mean(get_kl_lossfunc(mean, logstd, target_mu, target_logstd))
    rnn_comp = RNN_COMP_WITH_VAE(mean=mean, logstd=logstd, kl2vae=kl2vae,
                                 loss=rnn_loss, py=py, pz=pz, state_in=state_in,
                                 last_state=last_state)
    return rnn_comp

# TODO determine whether joint learning will be better.
def learn(sess, z_size, data_dir, max_seq_len,
          batch_size_per_task=16, rnn_size=256, kl_tolerance=0.5,
          vae_dir="tf_vae", model_dir="tf_rnn", target_dir="dream",
          layer_norm=False, fc_limit=False, rn=50,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0):
    # TODO remove this limit.
    n_tasks = 4
    batch_size = batch_size_per_task * n_tasks


    print("Batch size for each taks is", batch_size_per_task)
    print("The total batch size is", batch_size)

    check_dir(model_dir)
    lf = open(model_dir+'/log_%s' % datetime.now().isoformat(), "w")
    # define env
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    print("the environment", env_name, "has %i actions" % na)
    transform_types = ["Reconstruct", "Transpose", "Concat1", "Color"]
    wrappers = [None, DatasetTransposeWrapper, DatasetHorizontalConcatWrapper, DatasetColorWrapper]
    seq_len = max_seq_len

    tf_v_lr = tf.placeholder(tf.float32, shape=[]) # learn from reconstruction.

    vaes = []
    vae_comps = []
    for i in range(n_tasks):
        vae = ConvVAE(name="vae%i" % i,
                      z_size=z_size,
                      batch_size=(seq_len + 1) * batch_size_per_task)
        vae_comp = build_vae("vae%i" % i, vae, na, z_size, seq_len, tf_v_lr, kl_tolerance, fc_limit)
        vaes.append(vae)
        vae_comps.append(vae_comp)


    # calculate the transform loss here
    comp = vae_comps[0]
    transform_losses = []
    for i in range(1, n_tasks):
      ty = vaes[i].build_decoder(comp.z, reuse=True)
      tty = wrappers[i].transform(ty)
      transform_loss = -tf.reduce_sum(comp.x * tf.log(tty + 1e-8) +
                               (1. - comp.x) * (tf.log(1. - tty + 1e-8)), [1, 2, 3])
      transform_loss = tf.reduce_mean(transform_loss)
      transform_losses.append(transform_loss)

    vae_total_loss = tf.reduce_mean([comp.loss for comp in vae_comps])

    print("The all vaes have been created")



    # build one continuous RNN
    rnn = VRNN("rnn",
                 max_seq_len,  # 4 for the recent frames
                 input_size,
                 output_size,
                 batch_size_per_task,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp,
                 True)



    tf_r_lr = tf.placeholder(tf.float32, shape=[])
    # Just build the architecture.
    rnn_comp = build_rnn("rnn", rnn, na, z_size, batch_size_per_task, seq_len)
    rnn_lv_dict = rnn.get_linear_variables()

    print("The basic rnn has been built")

    # phase 2
    rnn_losses = []
    rnn_meta_losses = []
    rnn_logstds = []
    kl2vaes = []
    rnn_vcomps = []
    vae_meta_var_list = []

    for i in range(n_tasks):
        comp = vae_comps[i]
        vae = vaes[i]
        if fc_limit:
          vae_meta_var_list += comp.fc_var_list
        else:
          vae_meta_var_list += comp.var_list
        tmp_rnn_lv_dict = copy.copy(rnn_lv_dict)
        rnn_vcomp = build_rnn_with_vae(vae, rnn, tmp_rnn_lv_dict, comp, z_size,
                                        seq_len, batch_size_per_task, tf_r_lr, kl_tolerance)


        rnn_logstds.append(rnn_vcomp.logstd)
        rnn_losses.append(rnn_vcomp.loss)
        kl2vaes.append(rnn_vcomp.kl2vae)
        rnn_vcomps.append(rnn_vcomp)


    ptransform_losses = []
    comp = vae_comps[0]
    for i in range(1, n_tasks):
      py = vaes[i].build_decoder(rnn_vcomps[0].pz, reuse=True) # pz shape [None, 32]
      py = wrappers[i].transform(py)

      ty = tf.reshape(comp.x, (batch_size_per_task, seq_len+1, 64, 64, 1))[:, 1:, ...]
      ty = tf.reshape(ty, (-1, 64, 64, 1))

      ptransform_loss = -tf.reduce_sum(ty * tf.log(py + 1e-8) +
                               (1. - ty) * (tf.log(1. - py + 1e-8)), [1, 2, 3])
      ptransform_loss = tf.reduce_mean(ptransform_loss)
      ptransform_losses.append(ptransform_loss)


    print("RNN has been connected to each VAE")
    verified_rnn_var_list=  rnn.get_variables()
    rnn_all_vn_list = [v.name for v in verified_rnn_var_list]
    rnn_vn_list = [v.name for v in verified_rnn_var_list if 'Adam' not in v.name]

    # We are going to minimize this
    # TODO the gradients part include the VAE. (It should be computed in adapted version.)
    rnn_total_loss = tf.reduce_mean(rnn_losses)

    rnn_mean_logstd = tf.reduce_mean(rnn_logstds)
    vae_mean_logstd = tf.reduce_mean([comp.logstd for comp in vae_comps])
    kl2vae_mean = tf.reduce_mean(kl2vaes)

    tf_vae_losses = tuple([comp.loss for comp in vae_comps])
    tf_rnn_losses = tuple(rnn_losses)
    tf_t_losses = tuple(transform_losses)
    tf_pt_losses = tuple(ptransform_losses)

    # The model has been loaded
    sess.run(tf.global_variables_initializer())
    for i, comp in enumerate(vae_comps):
      loadFromFlat(comp.var_list, vae_dir+"/vae%i.p" % i)
    loadFromFlat(rnn_comp.var_list, vae_dir+"/rnn.p")

    # check_dir(target_dir)

    # Load the data, only one episode and dream from the start
    # Sample the data one by one
    filelist = os.listdir(data_dir)
    filelist = [f for f in filelist if '.npz' in f]
    fn = random.choice(filelist)
    print("the file name is", fn)
    data = np.load(os.path.join(data_dir, fn))
    obs = data["obs"]
    obs = np.expand_dims(obs, axis=-1)
    obs = obs.astype(np.float32) / 255.0

    # TODO replace the action by the policy
    actions = data["action"]
    oh_actions = onehot_actions(actions, na)

    n = len(obs)

    # start from the initalization
    h_init = np.zeros((batch_size_per_task, rnn_size), np.float32)
    c_init = np.zeros((batch_size_per_task, rnn_size), np.float32)
    last_states = [(h_init, c_init) for i in range(n_tasks)]

    # aggregate all the hidden states into a tuple
    tf_last_states = [comp.last_state for comp in rnn_vcomps]
    tf_last_states = tuple(tf_last_states)

    tf_pys = [comp.py for comp in rnn_vcomps]
    tf_pys = tuple(tf_pys)

    check_dir(target_dir)
    check_dir(target_dir + '/origin/')
    check_dir(target_dir + '/compare/')
    check_dir(target_dir + '/dream/')
    for t in transform_types:
        check_dir(target_dir+'/dream/'+t)
        check_dir(target_dir+'/compare/'+t)

    # The model will share the first several frames.
    for i in range(n):
      frame = obs[i].reshape(-1, 64, 64, 1)
      action = oh_actions[i:i+1][None]
      feed = {}
      for j in range(n_tasks):
        comp = vae_comps[j]
        w = wrappers[j]
        # real data or dream data
        if i < rn:
          if w is None:
            feed[comp.x] = frame
          else:
            feed[comp.x] = w.data_transform(frame)
        else:
            feed[comp.x] = pys[j]

        feed[comp.a] = action # the same for all tasks.
        rcomp = rnn_vcomps[j]
        feed[rcomp.state_in[0]] = last_states[j][0]
        feed[rcomp.state_in[1]] = last_states[j][1]

      last_states, pys = sess.run([tf_last_states, tf_pys] ,feed)
      # All of them are dream world
      for j, py in enumerate(pys):
          img = py.reshape(64,64)*255.
          imsave(target_dir+'/dream/'+transform_types[j]+'/%s.png' % pad_num(i), img)
          w = wrappers[j]
          if w is not None:
            img = w.data_transform(py)
            img = img.reshape(64,64)*255.
          imsave(target_dir+'/compare/'+transform_types[j]+'/%s.png' % pad_num(i), img)
      imsave(target_dir+'/origin/'+'%s.png' % pad_num(i), frame.reshape(64,64)*255.)
      



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
    parser.add_argument('--data-dir', default="record", help="the directory of data")
    parser.add_argument('--max-seq-len', type=int, default=1, help="the maximum steps of dynamics to catch")
    parser.add_argument('--batch-size-per-task', type=int, default=1, help="batch size for each task")

    parser.add_argument('--rnn-size', type=int, default=256, help="rnn hidden state size")


    # to load
    # Transfer the data directly
    # parser.add_argument('--n-updates', type=int, default=1, help="number of inner gradient updates during training")

    parser.add_argument('--vae-dir', default="tf_vae", help="the path of vae models to load")
    parser.add_argument('--target-dir', default="dream_world", help="the output of dream world")

    parser.add_argument('--kl-tolerance', type=float, default=0.5, help="kl tolerance")

    parser.add_argument('--model-dir', default="tf_rnn", help="the directory to store rnn model")
    parser.add_argument('--layer-norm', action="store_true", default=False, help="layer norm in RNN")
    parser.add_argument('--fc-limit', action="store_true", default=False, help="limit training the fc layers in vae")
    parser.add_argument('--recurrent-dp', type=float, default=1.0, help="dropout ratio in recurrent")
    parser.add_argument('--input-dp', type=float, default=1.0, help="dropout ratio in input")
    parser.add_argument('--output-dp', type=float, default=1.0, help="dropout ratio in output")
    parser.add_argument('--rn',  type=int, default=50, help="the first rn frames to be the real frames")
    args = vars(parser.parse_args())

    check_dir(args["model_dir"])
    with open(args["model_dir"]+'/args.json', "w") as f:
      json.dump(args, f, indent=2, sort_keys=True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        learn(sess, **args)


if __name__ == '__main__':
  main()

