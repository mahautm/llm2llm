"""
PPO implementation taken from https://github.com/openai/spinningup
"""

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# def wt_cumsum(t, wt):
#     # weighted cumulative sum with tensor operations
#     wts = wt ** torch.arange(t.shape[1], device=t.device)
#     return (torch.cumsum(wts * t.flip([1]), dim=1) / wts).flip([1])
def discount_cumsum(t, discount):
    wts = discount ** torch.arange(
        t.shape[1], dtype=torch.float64, device=t.device
    )  # pre-compute all discounts
    x = wts * t  # weighted vector
    cum_sum = torch.cumsum(x, dim=1)  # forward cumsum
    re_cum_sum = x - cum_sum + cum_sum[:, -1:]  # reversed cumsum
    return re_cum_sum / wts


def pad_merge(a, b, pad_value=0):
    # pad then concat
    _position = (
        (b.shape[1] - a.shape[1], 0)
        if len(a.shape) == 2
        else (0, 0, b.shape[1] - a.shape[1], 0)
    )
    return torch.cat(
        [
            torch.functional.F.pad(
                a,
                _position,
                "constant",
                pad_value,
            ),
            b,
        ],
    )


# def discount_cumsum(x, discount):
#     """
#     magic from rllab for computing discounted cumulative sums of vectors.
#     input:
#         vector x,
#         [x0,
#          x1,
#          x2]
#     output:
#         [x0 + discount * x1 + discount^2 * x2,
#          x1 + discount * x2,
#          x2]
#     """
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def scores_to_proba_dists(scores):
    proba_dists = []
    scores_max = torch.max(scores, dim=1)[0]
    # scores_max cannot be 0, otherwise the softmax will be NaN. We add a small value to avoid this.
    scores_max += 1e-8
    for j in range(len(scores)):
        proba_dists.append(
            F.softmax(scores[j] / scores_max[j], dim=-1).unsqueeze(dim=0)
        )

    proba_dists = torch.cat(proba_dists)
    dists = Categorical(probs=proba_dists)

    return dists
