import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
from scipy.misc import imsave
from utils import pad_num, loadFromFlat

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

DATA_DIR = "record"
model_path_name = "vae"
output_dir = "vae_test_result"

z_size=32

filelist = os.listdir(DATA_DIR)
filelist = [f for f in filelist if '.npz' in f]

filename =  random.choice(filelist)
print("the file name is", filename)
obs = np.load(os.path.join(DATA_DIR, filename))["obs"]
obs = np.expand_dims(obs, axis=-1)
obs = obs.astype(np.float32)/255.0

n = len(obs)

vae = ConvVAE(name="conv_vae",
              z_size=z_size,
              batch_size=1)

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="x")
z = vae.build_encoder(x)
y = vae.build_decoder(z)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

tf_r_loss = tf.reduce_mean(-tf.reduce_sum(x * tf.log(y+1e-8) +
                                      (1-x) * (tf.log(1-y+1e-8)),[1,2,3]))

loadFromFlat(vae.get_variables(), 'tf_vae/final_vae_transpose.p')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

r_losses = []
print(n, "images loaded")
for i in range(n):
    # input
    frame = obs[i].reshape(1, 64, 64, 1)
    frameT = np.transpose(frame, [0, 2, 1, 3])
    feed = {x: frameT}
    reconstructT, r_loss = sess.run([y, tf_r_loss], feed)
    #reconstruct, r_loss = sess.run([y, tf_r_loss], feed)
    reconstruct = np.transpose(reconstructT, [0, 2, 1, 3])
    r_losses.append(r_loss)
    #print(i, np.max(frame), np.max(reconstruct), r_loss)
    #imsave(output_dir+'/%s.png' % pad_num(i), 255.*frame[0].reshape(64, 64))
    imsave(output_dir+'/%s_vaet.png' % pad_num(i), 255.*reconstruct[0].reshape(64, 64))
print("the mean of reconstruction loss", np.mean(r_losses))
