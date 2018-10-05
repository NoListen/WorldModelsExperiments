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
from utils import create_vae_dataset, check_dir, saveToFlat, loadFromFlat, wrap, \
    onehot_actions, onehot_action, check_dir, get_lr_lossfunc,  get_kl_lossfunc, \
    get_kl2normal_lossfunc
from env import make_env
import os
from env import make_atari
from config import env_name
from collections import deque
from wrappers import DatasetTransposeWrapper, DatasetSwapWrapper, DatasetHorizontalConcatWrapper,\
                       DatasetVerticalConcatWrapper, DatasetColorWrapper
from mlp_policy import MlpPolicy
import tf_utils as U
from dataset import Dataset

VAE_COMP = namedtuple('VAE_COMP', ['a', 'x', 'y', 'z', 'mu', 'logstd', 'ma', 'mx', 'my', 'mz', 'mmu', 'mlogstd', 
                                    'r_loss', 'kl_loss', 'loss', 'var_list', 'fc_var_list', 'train_opt'])
RNN_COMP_WITH_OPT = namedtuple('RNN_COMP', ['z_input', 'a', 'logmix', 'mean', 'logstd', 'var_list'])
RNN_COMP_WITH_VAE = namedtuple("RNN_COMP_WITH_VAE", ['logstd', 'mean', 'loss', 'py',
                                                     'pz', 'kl2vae', 'state_in', 'last_state'])

np.random.seed(1234567)

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def traj_segment_generator(pi, env, horizon=100, stochastic=True):
    t = 0
    new = True
    ob = env.reset()

    obs = np.array([ob for _ in range(horizon)])
    # discrete simple action
    acs = np.zeros(horizon, "int32")
    news = np.zeros(horizon, "int32")
    vpreds = np.zeros(horizon, "float32")
    rews = np.zeros(horizon, "float32")

    ep_rets = []
    ep_lens = []
    cur_ep_ret = 0
    cur_ep_len = 0
    
    wid = int(np.random.rand() > 0.5)
    while True:
        #wid = int(np.random.rand() > 0.5)
        ac, vpred = pi.act(stochastic, ob[wid])
        if t % horizon == 0 and t > 0:
            yield{"ob": obs, "ac": acs, "rew": rews, "vpred": vpreds, "new": news,
                  "nextvpred": vpred * (1-new), "ep_rets": ep_rets, "ep_lens": ep_lens}
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        # pong's action need to be a tuple
        ob, rew, new = env.step(ac)
        cur_ep_ret += rew
        cur_ep_len += 1

        t += 1

        if new:
            ep_lens.append(cur_ep_len)
            ep_rets.append(cur_ep_ret)
            cur_ep_ret = 0
            cur_ep_len = 0
            wid = int(np.random.rand() > 0.5)
            ob = env.reset()

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    # I think is has been simplified
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

class PongModelWrapper(object):
    def __init__(self, sess, na, n_tasks, wrappers, rnn_size, vcomps, rcomps):
        self.sess = sess
        self.n_tasks = n_tasks
        self.na = na
        self.env = make_atari("PongNoFrameskip-v4", noop_max=3, clip_frame=True)
        h_init = np.zeros((1, rnn_size), np.float32)
        c_init = np.zeros((1, rnn_size), np.float32)
        self.init_states = [(c_init, h_init) for _ in range(n_tasks)]
        self.last_states = self.init_states

        self.vcomps = vcomps
        self.rcomps = rcomps
        self.wrappers = wrappers

        tf_last_states = [rcomp.last_state for rcomp in rcomps]
        self.tf_last_states = tuple(tf_last_states)

        #tf_z = [vcomp.z for vcomp in vcomps]
        #self.tf_z = tuple(tf_z)
        tf_mu = [vcomp.mu for vcomp in vcomps]
        self.tf_z = tuple(tf_mu)


        self.last_s = None

    def reset(self):
        s = self.env.reset()
        # reset. I think this won't be changed
        self.last_states = self.init_states
        feed = {}
        rs = s.reshape(1, 64, 64, 1)/255.
        for i in range(self.n_tasks):
            vcomp = self.vcomps[i]
            feed[vcomp.x] = wrap(self.wrappers[i], rs)
        self.last_s = rs
        zs = self.sess.run(self.tf_z, feed)
        zhs = self.get_zh(zs)
        return zhs

    def step(self, action):
        oh_action = onehot_action(action, self.na)
        oh_action = oh_action.reshape(1, 1, -1)
        s, r, done, _ = self.env.step(action)
        # Now we use the states generated by the RNN.
        feed = {}
        rs = s.reshape(1, 64, 64, 1)/255.
        for i in range(self.n_tasks):
            vcomp = self.vcomps[i]
            feed[vcomp.x] = wrap(self.wrappers[i], self.last_s)
            feed[vcomp.a] = oh_action

            rcomp = self.rcomps[i]
            last_state = self.last_states[i]
            feed[rcomp.state_in[0]] = last_state[0]
            feed[rcomp.state_in[1]] = last_state[1]
        # Update the hidden states
        self.last_states = self.sess.run(self.tf_last_states, feed)
        self.last_s = rs
        # TODO see the world model code

        for i in range(self.n_tasks):
            vcomp = self.vcomps[i]
            feed[vcomp.x] = wrap(self.wrappers[i], rs)
        zs = self.sess.run(self.tf_z, feed)
        zhs = self.get_zh(zs)
        return zhs, r, done

    def get_zh(self, zs):
        zhs = []
        for i in range(self.n_tasks):
            zh = np.concatenate([zs[i], self.last_states[i][0], self.last_states[i][1]], axis=1)
            zhs.append(zh)
        zhs = np.concatenate(zhs, axis=0)
        return zhs

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
    out_logmix, out_mean, out_logstd = rnn.build_model(input_x)

    rnn_var_list = rnn.get_variables()
    rnn_comp = RNN_COMP_WITH_OPT(a=a, z_input=rnn_z, logmix=out_logmix,
                                 mean=out_mean, logstd=out_logstd, 
                                var_list=rnn_var_list)

    return rnn_comp


# Do we need another placeholder ?

def process_z_with_vae(x, z, a, batch_size, seq_len, z_size):
    # reshape and cut
    target_y = tf.reshape(x, (batch_size, seq_len, 64, 64, 1))
    target_y = tf.reshape(target_y, (-1, 64, 64, 1))

    input_z = tf.reshape(z, (batch_size, seq_len, z_size))
    input_z = tf.concat([input_z, a], axis=2)

    return input_z, target_y


def rnn_with_vae(vae, rnn, x, z, a, z_size, batch_size, seq_len, kl_tolerance):
    input_z, target_y = process_z_with_vae(x, z, a, batch_size, seq_len, z_size)

    c_in = tf.placeholder(tf.float32, (batch_size, rnn.rnn_size))
    h_in = tf.placeholder(tf.float32, (batch_size, rnn.rnn_size))
    state_in = [c_in, h_in]
    pz, mean, logstd, last_state = rnn.build_cont_model(input_z, c_in, h_in, reuse=True)

    mean = tf.reshape(mean, [-1, z_size])
    logstd = tf.reshape(logstd, [-1, z_size])
    pz = tf.reshape(pz, [-1, z_size])
    py = vae.build_decoder(pz, reuse=True)  # -1, 64, 64, 1
    rnn_loss = tf.reduce_mean(get_lr_lossfunc(target_y, py))
    rnn_kl_loss = get_kl2normal_lossfunc(mean, logstd)
    rnn_loss += tf.reduce_mean(tf.maximum(rnn_kl_loss, kl_tolerance * z_size))
    return rnn_loss, mean, logstd, py, pz, state_in, last_state

def cut(x, batch_size, seq_len, z_size):
    x = tf.reshape(x, [batch_size, seq_len+1, z_size])[:, 1:, :]
    x = tf.reshape(x, [-1, z_size])
    return x


# Meta part.
def build_rnn_with_vae(vae, rnn, comp, z_size, seq_len, batch_size, kl_tolerance=0.5):
    rnn_loss, mean, logstd, py, pz, state_in, last_state = rnn_with_vae(vae, rnn, comp.x, comp.z, comp.a,
                                                    z_size, batch_size, seq_len, kl_tolerance)

    target_mu = cut(comp.mu, batch_size, seq_len, z_size)
    target_logstd = cut(comp.logstd, batch_size, seq_len, z_size)
    kl2vae = tf.reduce_mean(get_kl_lossfunc(mean, logstd, target_mu, target_logstd))
    rnn_comp = RNN_COMP_WITH_VAE(mean=mean, logstd=logstd, kl2vae=kl2vae,
                                 loss=rnn_loss, py=py, pz=pz, state_in=state_in,
                                 last_state=last_state)
    return rnn_comp

def build_world_model(sess, n_tasks, z_size, rnn_size=256,
          kl_tolerance=0.5, transform="transpose",
          model_dir="tf_rnn", layer_norm=False,
          recurrent_dp = 1.0,
          input_dp = 1.0,
          output_dp = 1.0, **kargs):
    # TODO remove this limit.
    batch_size_per_task = 1
    batch_size = batch_size_per_task * n_tasks
    print("Batch size for each taks is", batch_size_per_task)
    print("The total batch size is", batch_size)
    if transform == "transpose":
      wrapper = DatasetTransposeWrapper
    elif transform == "swap":
      wrapper = DatasetSwapWrapper
    elif transform == "concat1":
      wrapper = DatasetHorizontalConcatWrapper
    elif transform == "concat2":
      wrapper = DatasetVerticalConcatWrapper
    elif transform == "color":
      wrapper = DatasetColorWrapper
    else:
      raise Exception("Such transform is not available")
    wrappers = [None, wrapper]


    check_dir(model_dir)
    # define env
    na = make_env(env_name).action_space.n
    input_size = z_size + na
    output_size = z_size

    print("the environment", env_name, "has %i actions" % na)

    # build dataset
    seq_len = 1
    tf_v_lr = tf.placeholder(tf.float32, shape=[]) # learn from reconstruction.

    vaes = []
    vae_comps = []
    for i in range(n_tasks):
        vae = ConvVAE(name="vae%i" % i,
                      z_size=z_size,
                      batch_size= seq_len * batch_size_per_task)
        vae_comp = build_vae("vae%i" % i, vae, na, z_size, seq_len, tf_v_lr, kl_tolerance)
        vaes.append(vae)
        vae_comps.append(vae_comp)

    # Meta RNN.
    rnn = VRNN("rnn",
                 seq_len,  # 4 for the recent frames
                 input_size,
                 output_size,
                 batch_size_per_task,  # minibatch sizes
                 rnn_size,  # number of rnn cells
                 layer_norm,
                 recurrent_dp,
                 input_dp,
                 output_dp)

    # Just build the architecture.
    rnn_comp = build_rnn("rnn", rnn, na, z_size, batch_size_per_task, seq_len)

    print("The basic rnn has been built")

    rnn_vcomps = []

    for i in range(n_tasks):
        comp = vae_comps[i]
        vae = vaes[i]
        rnn_vcomp = build_rnn_with_vae(vae, rnn, comp, z_size,
                                       seq_len, batch_size_per_task, kl_tolerance)
        rnn_vcomps.append(rnn_vcomp)

    return na, wrappers, vae_comps, rnn_comp, rnn_vcomps

# pi is the agent
def ac(env, pi, gamma, lam, horizon, rl_dir):
    seg_gen = traj_segment_generator(pi, env, horizon, stochastic=True)

    ep_rets = deque(maxlen=100)
    ep_lens = deque(maxlen=100)
    last_episodes = episodes = 0

    while True:
        # generate one episode
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        # output the recent one.

        if len(seg["ep_rets"]) > 0:
            ep_rets.extend(seg["ep_rets"])
            ep_lens.extend(seg["ep_lens"])
            for i in range(len(seg["ep_rets"])):
                print("ep%i ret:%.2f steps:%i average_ret:%.1f average_steps:%.1f" %
                    (episodes+i+1, seg["ep_rets"][i], seg["ep_lens"][i], np.mean(ep_rets), np.mean(ep_lens)))
            episodes += len(seg["ep_rets"])

        #print(np.unique(seg["ac"]))
        #print("avg value %.2f" % np.mean(np.mean(seg["vpred"])))
        #print("avg advantage %.2f" % np.mean(np.mean(seg["adv"])))
        #print(seg["tdlamret"])
        #print(seg["new"])
        #print("##########################################################")
        pi.train(seg)
        if episodes - last_episodes > 100:
            last_episodes = episodes
            saveToFlat(pi.net.get_variables(), rl_dir+'/%i.p' % episodes)


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
    parser.add_argument('--rnn-size', type=int, default=256, help="rnn hidden state size")
    parser.add_argument('--transform', default="transpose", help="type of transform. ['transform', 'color', 'swap', 'concat1', 'concat2']")
    parser.add_argument('--n-tasks', type=int, default=2, help="the number of tasks")
    parser.add_argument('--kl-tolerance', type=float, default=0.5, help="kl tolerance")
    parser.add_argument('--model-dir', default="tf_rnn", help="the directory to store rnn model")
    parser.add_argument('--rl-dir', default="rl_mirror", help="the directory to store rl model")
    parser.add_argument('--layer-norm', action="store_true", default=False, help="layer norm in RNN")
    parser.add_argument('--recurrent-dp', type=float, default=1.0, help="dropout ratio in recurrent")
    parser.add_argument('--input-dp', type=float, default=1.0, help="dropout ratio in input")
    parser.add_argument('--output-dp', type=float, default=1.0, help="dropout ratio in output")

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--horizon', type=int, default=10000)

    parser.add_argument('--clip_param', type=float, default=0.2)
    args = vars(parser.parse_args())


    clip_param = args["clip_param"]
    check_dir(args["model_dir"])
    check_dir(args["rl_dir"])

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        na, wrappers, vcomps, rmcomp, rcomps = build_world_model(sess, **args)
        fsize = (args["z_size"] + args["rnn_size"] * 2,)
        pi = MlpPolicy("pi", input_size=fsize, num_output=na,
                       hid_size=16, num_hid_layers=2)
        oldpi = MlpPolicy("oldpi", input_size=fsize, num_output=na,
                          hid_size=16, num_hid_layers=2)

        tf_adv = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_adv')  # Empirical return
        tf_ret = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_ret')  # Empirical return
        tf_ac = pi.pdtype.sample_placeholder([None], name='tf_ac')

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-0.01) * meanent

        ratio = tf.exp(pi.pd.logp(tf_ac) - oldpi.pd.logp(tf_ac))
        surr1 = ratio * tf_adv
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * tf_adv
        policy_loss = - U.mean(tf.minimum(surr1, surr2))

        var_list = pi.get_variables()

        value_loss = tf.reduce_mean(tf.square(pi.vpred-tf_ret))
        total_loss = policy_loss + value_loss + pol_entpen
        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        opt_for_policy = tf.train.AdamOptimizer(learning_rate=args["lr"]).minimize(total_loss,
                                                                                   name='opt_p',
                                                                                   var_list=var_list)
        sess.run(tf.global_variables_initializer())

        # make up the environment
        for i, comp in enumerate(vcomps):
            loadFromFlat(comp.var_list, args["model_dir"] + "/vae%i.p" % i)
        loadFromFlat(rmcomp.var_list, args["model_dir"] + '/rnn.p')

        # I can ensure the env work properly
        env = PongModelWrapper(sess, na, args["n_tasks"], wrappers, args["rnn_size"],
                               vcomps, rcomps)

        steps = 0
        episodes = 1
        seg_gen = traj_segment_generator(pi, env, args["horizon"], stochastic=True)
        while True:
            steps += 1
            seg = seg_gen.__next__()
            print('\n')
            add_vtarg_and_adv(seg, args["gamma"], args["lam"])
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            atarg = (atarg - atarg.mean()) / atarg.std()

            ob = np.transpose(ob, [1, 0, 2])
            ob = ob.reshape(-1, ob.shape[-1])
            ac = np.tile(ac, [2, 1])
            atarg = np.tile(atarg, [2, 1])
            tdlamret = np.tile(tdlamret, [2, 1])

            for i in range(len(seg["ep_rets"])):
              print("episode %i: obtain reward %.2f with %i steps" % (episodes, seg["ep_rets"][i], seg["ep_lens"][i]))
              episodes += 1
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            assign_old_eq_new()
            for _ in range(5):
                for batch in d.iterate_once(500):
                    _ = sess.run(opt_for_policy, feed_dict={pi.ob: batch["ob"],
                                                 pi.stochastic: True,
                                                 oldpi.ob: batch["ob"],
                                                 oldpi.stochastic: True,
                                                 tf_ac: batch['ac'],
                                                 tf_ret: batch['vtarg'],
                                                 tf_adv: batch["atarg"]})
            if steps % 100 == 0:
              saveToFlat(var_list, args['rl_dir']+'/%i.p' % steps)

if __name__ == '__main__':
  main()

