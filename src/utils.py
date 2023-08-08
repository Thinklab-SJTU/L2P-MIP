import numpy as np
import torch
import torchvision
from easydict import EasyDict
import random

DOWNLOAD = True


def set_seed(seed=4358738):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def discrete2interval(val, bound, l, r):
    """Maps {0, 1, 2, ..., bound-1} to [l, r]."""
    return l + (r - l) / (bound - 1) * val


def discrete2bool(val, bound):
    """Maps {0, 1, 2, ..., bound-1} to {True, False}."""
    assert bound % 2 == 1  # fair
    return True if val <= bound // 2 else False


class CategoricalSampler:
    def __init__(self, probs):
        assert np.abs(np.sum(probs) - 1) < 1e-5
        assert len(probs) == 8
        self.probs = probs

    def sample(self):
        x = np.random.rand()
        for i, elem in enumerate(self.probs):
            if x < elem:
                return i
        return len(self.probs) - 1
