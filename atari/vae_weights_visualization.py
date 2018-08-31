import tensorflow as tf
from vae.vae import ConvVAE
from utils import loadFromFlat
import numpy as np
import pickle

def transpose_enc_fc(input):
  res = input.reshape((2,2,256,-1))
  res = np.transpose(res, [1, 0, 2, 3])
  return res

def alter_enc_weight(input ,z_size, order=None):
  res = input.reshape((-1, z_size))
  if order is None:
    order = np.arange(z_size)
    np.random.shuffle(order)
  res = res[:, order]
  print("shuffle in this order", order)
  return res, order

def alter_dec_weight(input, z_size, order):
  res = input.reshape((z_size, -1))
  res = res[order, :]
  print("shuffle in this order", order)
  return res

def transpose_dec_fc(input):
  res = input.reshape((-1,2,2,256))
  res = np.transpose(res, [0, 2, 1, 3])
  return res

def transpose_conv(input):
  res = np.transpose(input, [1, 0, 2, 3])
  return res

vae = ConvVAE(name="conv_vae",
                z_size=32,
                batch_size=1)

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="x")
mu, logstd, z = vae.build_encoder(x)
y = vae.build_decoder(z)
print(z.shape)
var_list = vae.get_variables()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
loadFromFlat(var_list, 'tf_vae/vae0.p')


var_values_list = []

order = None

for var in var_list:
  #print(var.name, var.shape)
  v = sess.run(var)
  l_type, w_type = var.name.split('/')[-2:]


  if 'conv' in l_type:
    if 'kernel' in w_type:
      print("conv rotate", var.name)
      #v = transpose_conv(v)
    else:
      print("do nothing")
  elif 'fc' in l_type:
    if 'enc' in l_type:
        if 'kernel' in w_type:
          pass
        #v, order = alter_enc_weight(v, 32, order)
    elif 'dec' in l_type:
       if 'kernel' in w_type:
         print(v.shape)
         print(v[9])
         print(np.sum(np.abs(v), axis=1))
         pass
         #v = alter_dec_weight(v, 32, order)
    else:
       print("error")
  else:
    print("error")

  #var_values_list.append(v.flatten())

#var_values_list = np.concatenate(var_values_list)
#print(var_values_list.shape)
#pickle.dump(var_values_list, open("tf_vae/vae1.p", "wb"))
