import numpy as np
from scipy.misc import logsumexp

logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))


def lognormal(y, mean, logstd):
  return -0.5 * ((y - mean) / np.exp(logstd)) ** 2 - logstd - logSqrtTwoPI


def neg_likelihood(logmix, mean, logstd, y):
  v = logmix + lognormal(y, mean, logstd)
  v = logsumexp(v, 1, keepdims=True)
  return -np.mean(v)


def onehot_actions(actions, na):
  actions = actions.astype(np.uint8)
  l = len(actions)
  oh_actions = np.zeros((l, na))
  oh_actions[np.arange(l), actions] = 1
  return oh_actions


def pad_num(n):
    s = str(n)
    return '0'*(4-len(s)) + s

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1


def sample_z(logmix, mean, logstd, l, T = 1):
    if T == 1:
        logmix2 = np.copy(logmix)/T
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(l, 1)
    else:
        logmix2 = np.copy(logmix)

    mixture_idx = np.zeros(l)
    chosen_mean = np.zeros(l)
    chosen_logstd = np.zeros(l)

    for j in range(l):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(l) * np.sqrt(T)
    next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
    return next_z