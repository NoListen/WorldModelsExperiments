import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
from scipy.misc import imsave
from utils import pad_num, loadFromFlat
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

DATA_DIR = "record"
model_dir = "swap/swap3"
output_dir = "vae_swap_result"

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
_, _, z = vae.build_encoder(x)
y = vae.build_decoder(z)
var_list = vae.get_variables()

vae2 = ConvVAE(name="conv_vae2",
              z_size=z_size,
              batch_size=1)

x2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="x2")
_, _, z2 = vae2.build_encoder(x2)
y2 = vae2.build_decoder(z2)

var_list2 = vae2.get_variables()

my = vae2.build_decoder(z, reuse=True)
#xt = tf.transpose(x, [0, 2, 1, 3])
xt = x[:, :, ::-1, :]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

tf_r_loss = tf.reduce_mean(-tf.reduce_sum(xt * tf.log(my+1e-8) +
                                      (1-xt) * (tf.log(1-my+1e-8)),[1,2,3]))

tf_r_loss_self = tf.reduce_mean(-tf.reduce_sum(x * tf.log(my+1e-8) +
                                      (1-x) * (tf.log(1-my+1e-8)),[1,2,3]))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

r_losses = []
print(n, "images loaded")

dns = os.listdir(model_dir)
dns = [dn for dn in dns if 'it' in dn]

ids = [int(dn[3:]) for dn in dns]
max_id = np.max(ids)

hist = []

obs = obs.reshape(-1, 64, 64, 1)
for i in range(max_id//10 + 1):
    dn = model_dir + '/it_' + str(i*10)
    loadFromFlat(var_list, dn+"/vae0.p")
    loadFromFlat(var_list2, dn+"/vae1.p")
    feed = {x: obs}
    loss = sess.run([tf_r_loss, tf_r_loss_self], feed)
    r_losses.append(loss)
    print(i, "r_loss: %.4f r_loss_self: %.4f" % (loss[0], loss[1]))
with open("swap_loss.p" , "wb") as f:
  pickle.dump(r_losses , f)
