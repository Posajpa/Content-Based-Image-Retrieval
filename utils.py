import numpy as np
import random
import torch

def seed_everything(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)