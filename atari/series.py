'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
import argparse
from env import make_env
from utils import onehot_actions

def count_min_length_of_filelist(filelist):
  N = len(filelist)
  min_length = 1000
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['action']
    l = len(raw_data)
    min_length = min(min_length, l)
  return  min_length

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Pong-v0')
args = parser.parse_args()

# number of actions
N = make_env(args.env).action_space.n
print("environment", args.env, "has", N, "discrete actions")

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"
ENV_NAME = "PongNoFrameskip-v0"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist, min_seq_len):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    obs = raw_data['obs'][:min_seq_len]
    action = raw_data['action'][:min_seq_len, ...]

    obs = np.expand_dims(obs, axis=-1)
    oh_action = onehot_actions(action, N) # N is a global variable
    data_list.append(obs)
    action_list.append(oh_action)
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode_batch(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(batch_size, 64, 64, 1)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 1)
  return batch_img


# Hyperparameters for ConvVAE
z_size=32
# batch_size is determined by min seq length
#batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
# eliminate other files.
filelist = [f for f in filelist if '.npz' in f]
filelist = filelist[0:10000]

min_seq_len = count_min_length_of_filelist(filelist)
batch_size = min_seq_len
print("Min Seq length is", min_seq_len)
dataset, action_dataset = load_raw_data_list(filelist, min_seq_len)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
for i in range(len(dataset)):
  data_batch = dataset[i]
  mu, logvar, z = encode_batch(data_batch)
  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  if ((i+1) % 1000 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

print("The action shape is", action_dataset.shape)
print("The mu shape is", mu_dataset.shape)
print("The log var shape is", logvar_dataset.shape)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
