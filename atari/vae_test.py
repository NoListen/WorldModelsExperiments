import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
from scipy.misc import imsave
from utils import pad_num

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

DATA_DIR = "record"
model_path_name = "vae"
output_dir = "vae_test_result"

z_size=32

filelist = os.listdir(DATA_DIR)
filelist = [f for f in filelist if '.npz' in f]

obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
obs = np.expand_dims(obs, axis=-1)
obs = obs.astype(np.float32)/255.0

n = len(obs)

vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, 'vae.json'))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(n, "images loaded")
for i in range(n):
    frame = obs[i].reshape(1, 64, 64, 1)
    batch_z = vae.encode(frame)
    reconstruct = vae.decode(batch_z)
    imsave(output_dir+'/%s.png' % pad_num(i), 255.*frame[0].reshape(64, 64))
    imsave(output_dir+'/%s_vae.png' % pad_num(i), 255.*reconstruct[0].reshape(64, 64))