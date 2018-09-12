import argparse
import pickle
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--N', help="the first N points", type=int, default=0)
args = parser.parse_args()
from numpy import log10 as log

font = {'size': 24}
import matplotlib
matplotlib.rc('font', **font)
COLOR = ['r', 'y', 'c', 'm']

def plot(r, N, k, world_types):
  t = np.arange(N) * 200
  fig, ax = plt.subplots()
  plt.gcf().subplots_adjust(bottom=0.23)
  plt.gcf().subplots_adjust(left=0.15)
  ax.xaxis.set_major_locator(plt.MaxNLocator(2))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3))
  if k in ["rnn_cost", "vae_cost"]:
    sub = 16
  else:
    sub = 0

  for i, w in enumerate(world_types):
    ax.plot(t, log(np.array(r[k][:N, i]) - sub), COLOR[i], label=w)

  ax.set_xlabel("time step")
  ax.set_ylabel("log %s loss" % (k.split('_')[0]))
  ax.set_ylim(1, 3.5)
  matplotlib.rcParams.update({'font.size': 24})
  ax.legend(loc=1)
  plt.savefig("all_%s.pdf" % k)

import json
import numpy as np
f = open("all.p", "rb")
r = pickle.load(f)
import matplotlib
import matplotlib.pyplot as plt

types = [r"$\Gamma_o$", r"$\Gamma_t$", r"$\Gamma_m$", r"$\Gamma_c$"]
if args.N > 0:
  N = min(args.N, len(r["rnn_cost"]))
else:
  N = len(r["rnn_cost"])

ks = ['rnn_cost', 'vae_cost', 'ptransform_cost', 'transform_cost']
for k, v in r.items():
  r[k] = np.array(v)

for k in ks:
  l = r[k].shape[1]
  plot(r, N, k, types[-l:])

f.close()
