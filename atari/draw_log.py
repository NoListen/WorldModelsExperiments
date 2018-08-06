import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', help='target history file', default='')
parser.add_argument('--sp', help='select points every sp points', type=int, default=1)
args = parser.parse_args()

 
#(t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)
if args.target == '':
  print("No selected file")
else:
  import json
  import numpy as np
  f = open("log/"+args.target, "r")
  r = json.load(f)
  # focus on avg_rewars
  r = np.array(r)
  import matplotlib
  matplotlib.use('Agg') # in case, DISPLAY is not defined
  import matplotlib.pyplot as plt
  plt.plot(r[::args.sp, 0], r[::args.sp, 2])
  plt.xlabel("time")
  plt.ylabel("avg reward")
  plt.savefig(args.target.split('.')[0]+".pdf")
  f.close()
