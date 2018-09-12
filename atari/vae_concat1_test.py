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
#model_path_name = "tf_vae"
#model_path_name = "tf_rnn/tmp"
model_path_name = "practice/d3/concat0b/it_2980"
output_dir = "result/vae_concat1_result"

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
xt = tf.split(x, 2,axis=2)
xt = tf.concat(xt[::-1], axis=2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

tf_r_loss = tf.reduce_mean(-tf.reduce_sum(xt * tf.log(my+1e-8) +
                                      (1-xt) * (tf.log(1-my+1e-8)),[1,2,3]))

print(var_list)
print(var_list2)

#loadFromFlat(var_list, model_path_name + '/final_vae0.p')
loadFromFlat(var_list, model_path_name+'/vae0.p')
#loadFromFlat(var_list2, model_path_name + '/final_vae1.p')
#loadFromFlat(var_list2, 'tf_rnn/final_vae1.p')
loadFromFlat(var_list2, model_path_name+'/vae1.p')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

r_losses = []
print(n, "images loaded")
for i in range(n):
    # input
    frame = obs[i].reshape(1, 64, 64, 1)
    #frameT = np.transpose(frame, [0, 2, 1, 3])
    feed = {x: frame}
    reconstruct, r_loss = sess.run([my, tf_r_loss], feed)
    #reconstruct = np.split(reconstructT, 2, axis=2)
    #reconstruct = np.concatenate(reconstruct[::-1], axis=2)
    r_losses.append(r_loss)
    #print(i, np.max(frame), np.max(reconstruct), r_loss)
    imsave(output_dir+'/%s.png' % pad_num(i), 255.*frame[0].reshape(64, 64))
    imsave(output_dir+'/%s_vae.png' % pad_num(i), 255.*reconstruct[0].reshape(64, 64))
print("the mean of reconstruction loss", np.mean(r_losses))
