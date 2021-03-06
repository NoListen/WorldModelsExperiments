'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
from scipy.misc import imsave
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph
import pickle

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5


# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"
LOAD_DATA = True
ENV_NAME = "BoxingNoFrameskip-v4"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=5000, M=1000): # N is 10000 episodes, M is number of timesteps
  data = np.zeros((M*N, 64, 64, 1), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    raw_data = np.expand_dims(raw_data, axis=-1)
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)

  if len(data) == M*N and idx < M*N:
    data = data[:idx]
  return data

if not os.path.exists("vae/data.p"):
  print("No data to load directly")
  LOAD_DATA = False

# load dataset from record/*. only use first 10K, sorted by filename.
if LOAD_DATA:
  with open("vae/data.p", "rb") as f:
    dataset = pickle.load(f)
else:
  filelist = os.listdir(DATA_DIR)
  filelist.sort()
  filelist = [f for f in filelist if '.npz' in f]
  filelist = filelist[0:10000]
  dataset = create_dataset(filelist)
  with open("vae/data.p", "wb") as f:
    pickle.dump(dataset, f, protocol=4)

print("The dataset has shape", dataset.shape)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

import os
if not os.path.exists("vimgs"):
  os.mkdir("vimgs")

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0
    feed = {vae.x: obs,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0 or train_step == 1):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      #rimgs = vae.sess.run(vae.y, feed)
      #tdir = "vimgs/%i" % train_step
      #if not os.path.exists(tdir):
      #  os.mkdir(tdir)
      #for i in range(len(rimgs)):
      #  imsave(tdir+'/%i.png' % i, rimgs[i].reshape(64,64))
      vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")
