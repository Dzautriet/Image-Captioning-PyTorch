# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:33:19 2020

@author: Zhe Cao
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        

def mask_accuracy(pred, targets, ignore_index):
    """
    pred: logit output
    target: labels
    ignore_index: exclude <pad> when calculating accuracy
    """
    mask = ~targets.eq(ignore_index).cuda()
    pred = pred[mask]
    targets = targets[mask]
    num_correct = pred.argmax(dim=1).eq(targets).sum()
    acc = num_correct.float() / targets.size(0)
    return acc

