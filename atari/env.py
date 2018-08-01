import numpy as np
import gym
from gym import spaces

from scipy.misc import imresize as resize
from gym.spaces.box import Box
import cv2
from scipy.misc import imsave
import time
SCREEN_X = 64
SCREEN_Y = 64


class PongBinary(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = frame[35:195, :, 0]
        frame[frame==144] = 0
        frame[frame==109] = 0
        frame[frame!=0] = 255
        frame = frame[::2, ::2]
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        return frame

# Borrowed from the universe-starter-agent, openai baselines

class AtariRescale64x64(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        return frame


class AtariRescaleClip64x64(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (64, 84), interpolation=cv2.INTER_AREA)
        frame = frame[14:78, ...]
        return frame


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def make_atari(env_id, noop_max=3, clip_frame=True):
  env = gym.make(env_id)
  env = NoopResetEnv(env, noop_max=noop_max)
  env = SkipEnv(env, skip=4)
  # env = MaxAndSkipEnv(env, skip=4)
  if env_id == "PongNoFrameskip-v4":
      env = PongBinary(env)
  else:
      if clip_frame:
          env = AtariRescaleClip64x64(env)
      else:
          env = AtariRescale64x64(env)
  env = ClipRewardEnv(env)
  return env

  # useless render mode
def make_env(env_name, seed=-1, render_mode=False):
  env = make_atari(env_name)
  if (seed >= 0):
    env.seed(seed)


  # print("environment details")
  # print("env.action_space", env.action_space)
  # print("high, low", env.action_space.high, env.action_space.low)
  # print("environment details")
  # print("env.observation_space", env.observation_space)
  # print("high, low", env.observation_space.high, env.observation_space.low)
  # assert False

  return env

# from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
# if __name__=="__main__":
#   from pyglet.window import key
#   a = np.array( [0.0, 0.0, 0.0] )
#   def key_press(k, mod):
#     global restart
#     if k==0xff0d: restart = True
#     if k==key.LEFT:  a[0] = -1.0
#     if k==key.RIGHT: a[0] = +1.0
#     if k==key.UP:    a[1] = +1.0
#     if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
#   def key_release(k, mod):
#     if k==key.LEFT  and a[0]==-1.0: a[0] = 0
#     if k==key.RIGHT and a[0]==+1.0: a[0] = 0
#     if k==key.UP:    a[1] = 0
#     if k==key.DOWN:  a[2] = 0
#   env = CarRacing()
#   env.render()
#   env.viewer.window.on_key_press = key_press
#   env.viewer.window.on_key_release = key_release
#   while True:
#     env.reset()
#     total_reward = 0.0
#     steps = 0
#     restart = False
#     while True:
#       s, r, done, info = env.step(a)
#       total_reward += r
#       if steps % 200 == 0 or done:
#         print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
#         print("step {} total_reward {:+0.2f}".format(steps, total_reward))
#       steps += 1
#       env.render()
#       if done or restart: break
#   env.monitor.close()
