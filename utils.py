from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import pdb
import torch

import numpy as np

import torch.nn.functional as F
import sys
import pandas as pd

import math


def chi2_distance(A, B):
 
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                      for (a, b) in zip(A, B)])
    return chi



    
def Cmc(qf,gf,q_pids, g_pids, q_camids, g_camids, rank_size):
    qf = qf.numpy()
    gf = gf.numpy()
    n_query = qf.shape[0]
    n_gallery = gf.shape[0]

    dist = np_cdist(qf, gf) # Reture a n_query*n_gallery array

    cmc = np.zeros((n_query, rank_size))
    ap = np.zeros(n_query)
    
    #widgets = ["I'm calculating cmc! ", AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' (', Percentage(), ')']
    #pbar = ProgressBar(widgets=widgets, max_value=n_query)
    for k in range(n_query):
        good_idx = np.where((q_pids[k]==g_pids) & (q_camids[k]!=g_camids))[0]
        junk_mask1 = (g_pids == -1)
        junk_mask2 = (q_pids[k]==g_pids) & (q_camids[k]==g_camids)
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        score = dist[k, :]
        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:rank_size]
        pdb.set_trace()
        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)
#        pbar.update(k)
   # pbar.finish()
    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP

def Compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    pdb.set_trace()
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n-njunk:] = 1
            flag = 1 # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue # junk image
        
        if flag == 1:
            intersect_size += 1
        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap += (recall-old_recall) * (old_precision+precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1
       
        if good_now == ngood:
            return ap, cmc
    return ap, cmc


#def emd_np(a, b):
 #   ac = np.cumsum(a)
  #  bc = np.cumsum(b)
   # return np.sum(np.abs(ac-bc))

def emd_np(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)




def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)





def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))








