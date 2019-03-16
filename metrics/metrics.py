
'''
Code snippets for keeping track of evaluation metrics
'''

import numpy as np
import json

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,empty = False):
        if not(empty):
            self.reset()
        else:
            self.update = lambda val,n=1 : None
            self.value = lambda : -1

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

    def value(self):
        return self.avg

class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self,empty = False):
        if not(empty):
            self.reset()
        else:
            self.update = lambda val,n=1 : None
            self.value = lambda : -1

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum


class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,empty = False):
        if not(empty):
            self.reset()
        else:
            self.update = lambda val : None
            self.value = lambda : -1

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val


def make_meters(empty=False):
    meters_dict = {
        'style_loss': ValueMeter(empty=empty),
        'content_loss': ValueMeter(empty=empty),
        'reg_loss': ValueMeter(empty=empty),
        'tv_loss': ValueMeter(empty=empty),
        'total_loss': ValueMeter(empty=empty),
        'epoch_time': ValueMeter(empty=empty),
        'lr' : ValueMeter(empty=empty)
    }
    return meters_dict



def save_meters(meters, fn, epoch=0):

    logged = {}
    for name, meter in meters.items():
        logged[name] = meter.value()

    if epoch > 0:
        logged['epoch'] = epoch

    print(f'Saving meters to {fn}')
    with open(fn, 'w') as f:
        json.dump(logged, f)