"""
PPO implementation taken from https://github.com/openai/spinningup
"""
import numpy as np
import torch
from llm2llm.PPO_finetuning.utils.tools import discount_cumsum, combined_shape


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
        self.val_buf = np.zeros(size, dtype=np.float32)  # [None for _ in range(size)]
        self.logp_buf = [None for _ in range(size)]  # np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        print(f"Buffer size: {size}", "adv_buf size:", len(self.adv_buf))
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

    def store_v2(self, obs, rew, val, logp):
        """
        deals with arrays of difference sizes caused by rewards and obs only changing once per episode
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        _range = [i for i in range(self.ptr, self.ptr + len(val))]

        self.obs_buf[self.ptr] = obs
        self.rew_buf[_range] = 0
        self.rew_buf[self.ptr] = rew
        self.val_buf[_range] = val
        self.logp_buf[_range] = logp
        self.ptr += len(val)

    def finish_logit_path(self, batch_size, seq_len):
        # what I could do is add paths, AS IF the batch was bigger
        for i in range(batch_size):
            # determine path_slice made of two sequences of length seq_len, for each batch
            path_slice = np.concatenate(
                [
                    np.array([i for i in range(seq_len)])
                    + i * seq_len
                    + j * seq_len * batch_size
                    for j in range(
                        (self.ptr - self.path_start_idx) // (seq_len * batch_size)
                    )
                ],
                axis=0,
            )
            # I have doubts on the repetition of the last value
            # TODO: check GAE lambda implementation
            # print(path_slice)
            # print(self.rew_buf, self.val_buf)
            # print(len(self.adv_buf))
            rews = np.append(self.rew_buf[path_slice], self.val_buf[path_slice][-1])
            vals = np.append(self.val_buf[path_slice], self.val_buf[path_slice][-1])

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def finish_path(self, batch_size, expand=False):
        # what I could do is add paths, AS IF the batch was bigger
        # Added expansion for per logit rewards. the expand parameter should be the number of logits

        for i in range(batch_size):
            path_slice = [
                self.path_start_idx + i + j * batch_size
                for j in range((self.ptr - self.path_start_idx) // batch_size)
            ]
            rews = self.rew_buf[path_slice]
            vals = self.val_buf[path_slice]

            # the next two lines implement GAE-Lambda advantage calculation

            if expand:
                print(rews, vals.flatten(), len(vals.flatten()) // len(rews) - 1)
                rews = np.expand_dims(rews, axis=1)
                rews = np.pad(
                    rews,
                    ((0, 0), (0, len(vals.flatten()) // len(rews) - 1)),
                    "constant",
                    constant_values=0,
                ).flatten()
                print(rews, vals.flatten())
                _v = vals.flatten().numpy()
                deltas = rews[:-1] + self.gamma * _v[1:] - _v[:-1]
            else:
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def legacy_finish_path(self, last_vals):
        """
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

        (in this experiment there should be no epoch cutoffs)
        Added padding for logp_buf
        """
        # last_vals size should be batch_size
        batch_size = len(last_vals)
        for i in range(batch_size):
            path_slice = [
                self.path_start_idx + i + j * batch_size
                for j in range((self.ptr - self.path_start_idx) // batch_size)
            ]
            rews = np.append(self.rew_buf[path_slice], last_vals[i])
            vals = np.append(self.val_buf[path_slice], last_vals[i])

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

            # check max_seq_len
            # max_seq_len = max(max_seq_len, max([len(self.logp_buf[_idx]) for _idx in path_slice]))

        # # padding logp_buf
        # print('max_seq_len: ', max_seq_len)
        # self.logp_buf = torch.stack([torch.nn.functional.pad(_lp, (0, 0, 0, max_seq_len - _lp.shape[0]), value=1) for _lp in self.logp_buf])
        # print('logp_buf shape: ', self.logp_buf.shape)

        self.path_start_idx = self.ptr

    def to_txt(self, filename):
        """
        Adds the buffer to a text file.
        """
        data = {
            "obs": self.obs_buf,
            "rew": self.rew_buf,
            "val": self.val_buf,
            "ret": self.ret_buf,
            "adv": self.adv_buf,
            "logp": self.logp_buf,
        }
        with open(filename, "a") as f:
            for k, v in data.items():
                f.write(f"{k}: {v}\n")

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf[: self.ptr]), np.std(
            self.adv_buf[: self.ptr]
        )
        _adv_buf = (
            (self.adv_buf[: self.ptr] - adv_mean) / adv_std
            if adv_std != 0
            else self.adv_buf[: self.ptr]
        )
        data = dict(
            obs=self.obs_buf[: self.ptr],
            ret=self.ret_buf[: self.ptr],
            adv=_adv_buf,
            logp=self.logp_buf[: self.ptr],
        )
        self.ptr, self.path_start_idx = 0, 0
        return {
            k: torch.as_tensor(v, dtype=torch.float32) if not isinstance(v, list) else v
            for k, v in data.items()
        }
