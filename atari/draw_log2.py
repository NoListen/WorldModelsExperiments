import argparse
import pickle
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', help='target history file', default='')
parser.add_argument('--N', help="the first N points", type=int, default=0)
args = parser.parse_args()
from numpy import log10 as log

font = {'size': 32}
import matplotlib
matplotlib.rc('font', **font)

#(t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)
if args.target == '':
  print("No selected file")
else:
  import json
  import numpy as np
  f = open(args.target, "rb")
  r = pickle.load(f)
  # focus on avg_rewars
  import matplotlib
  # matplotlib.use('Agg') # in case, DISPLAY is not defined
  import matplotlib.pyplot as plt
  print(r.keys())
  if args.N > 0:
    N = min(args.N, len(r["rnn_cost"]))
  else:
    N = len(r["rnn_cost"])
  print(N)


  t = np.arange(N)*200
  print(log(np.array(r["transform_cost"][:N])))
  print(log(np.array(r["vae_cost"][:N])-16))
  fig, ax = plt.subplots()
  plt.gcf().subplots_adjust(bottom=0.23)
  plt.gcf().subplots_adjust(left=0.15)
  ax.xaxis.set_major_locator(plt.MaxNLocator(2))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3))
  ax.plot(t, log(np.array(r["rnn_cost"][:N])-16), 'r', label=r"$p$")
  ax.plot(t, log(np.array(r["vae_cost"][:N])-16), 'y', label=r"$r$")
  ax.plot(t, log(np.array(r["transform_cost"][:N])), 'c', label=r"$t$")
  ax.plot(t, log(np.array(r["ptransform_cost"][:N])), 'm', label=r"${pt}$")
  ax.set_xlabel("time step")
  ax.set_ylabel("log loss")
  ax.set_ylim(1, 3.5)
  matplotlib.rcParams.update({'font.size': 24})
  ax.legend(loc=1)
  plt.savefig(args.target.split('.')[0]+".pdf")
  f.close()
