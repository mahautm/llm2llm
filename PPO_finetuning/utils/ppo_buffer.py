'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import numpy as np
import torch

from . import discount_cumsum, combined_shape

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.obs_buf = [None for _ in range(size)]
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_vals):
        """
        # HERE we do not use this function : there should be no epoch cutoff
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # last_vals size should be batch_size
        batch_size = len(last_vals)
        for i in range(batch_size):
            path_slice = [self.path_start_idx + i + j*batch_size for j in range((self.ptr - self.path_start_idx)//batch_size)]
            rews = np.append(self.rew_buf[path_slice], last_vals[i])
            vals = np.append(self.val_buf[path_slice], last_vals[i])

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def to_txt(self, filename):
        """
        Adds the buffer to a text file.
        """
        data = {
            'obs': self.obs_buf,
            'rew': self.rew_buf,
            'val': self.val_buf,
            'ret': self.ret_buf,
            'adv': self.adv_buf,
            'logp': self.logp_buf,
        }
        with open(filename, 'a') as f:
            for k, v in data.items():
                f.write(f'{k}: {v}\n')

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }