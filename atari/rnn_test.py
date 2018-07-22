'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams, MDNRNN, hps_sample, rnn_next_state, rnn_init_state
from scipy.misc import imsave
from utils import pad_num, sample_z, neg_likelihood, onehot_actions

from env import make_env
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Pong-v0')
args = parser.parse_args()

# Disable the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
z_size=32
# temperature
T = 1.

# First, build the environment.
env = make_env(args.env)
na = env.action_space.n
print("environment", args.env, "has", na, "discrete actions")


# Second, read the data and make the directory.
DATA_DIR = "record"
output_dir = "rnn_test_result"

filelist = os.listdir(DATA_DIR)
filelist = [f for f in filelist if '.npz' in f]
file = random.choice(filelist)
raw_data = np.load(os.path.join(DATA_DIR, file))
obs = np.expand_dims(raw_data["obs"], axis=-1) # N X 64 X 64 X 1
obs = obs.astype(np.float32)/255.0
oh_actions = onehot_actions(raw_data["action"], na)
steps = len(obs)
print("The episode in file", file, "has length", steps)


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
reset_graph()

# Third, Build the VAE
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join('vae', 'vae.json'))


# Fourth, build the RNN
hps_atari_sample = hps_sample._replace(input_seq_width=z_size+na)
OUTWIDTH = hps_atari_sample.output_seq_width
rnn = MDNRNN(hps_atari_sample, gpu_mode=False)
rnn.load_json(os.path.join('rnn', 'rnn.json'))


print("All model loaded.")
# Fifth, run the evaluation. -> We have no predictions about the first frame.

start = time.time()

state = rnn_init_state(rnn) # initialize the state.
pz = None

for i in range(steps):

  ob = obs[i:i+1] # (1, 64, 64, 1)
  action = oh_actions[i:i+1] # (1, n)

  z = vae.encode(ob) # (1, 32) VAE done!
  rnn_z = np.expand_dims(z, axis=0) # (1, 1, 32)
  action = np.expand_dims(action, axis=0) # (1, 1, n)


  input_x = np.concatenate([rnn_z, action], axis=2) # (1, 1, 32+n)
  feed = {rnn.input_x: input_x, rnn.initial_state: state} # predict the next state and next z.

  if pz is not None: # decode from the z
    frame = vae.decode(pz[None])
    neglogp = neg_likelihood(logmix, mean, logstd, z.reshape(32,1))
    imsave(output_dir + '/%s_origin_%.2f.png' % (pad_num(i), np.exp(-neglogp)), 255.*ob.reshape(64, 64))
    # imsave(output_dir + '/%s_origin.png' % pad_num(i), 255.*ob.reshape(64, 64))
    imsave(output_dir + '/%s_reconstruct.png' % pad_num(i), 255. * frame[0].reshape(64, 64))


  (logmix, mean, logstd, state) = rnn.sess.run([rnn.out_logmix, rnn.out_mean,
                                                rnn.out_logstd, rnn.final_state], feed)



  # Sample the next frame's state.
  pz = sample_z(logmix, mean, logstd, OUTWIDTH, T)
