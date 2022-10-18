import torch
import random
import numpy as np
import os


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def decimal_to_string(dec):
    str_old = str(dec)
    str_new = ""
    for i in str_old:
        if i != ".":
            str_new += i
    return str_new

def sparser2coarser(targets, coarser_labels):
    # this is the list of the supergorup to which each class belong (so class 1 belong to superclass 4, classe 2 to superclass 1 and so on)
    return coarser_labels[targets]


def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]