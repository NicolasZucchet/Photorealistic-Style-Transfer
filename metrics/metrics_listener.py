
'''
listener class for storing intermediate parameters for Listeners
'''
import copy
import time
import json
import os
import shutil
from collections import defaultdict
from .metrics import make_meters

'''
Object of the Listener class keep track of scores and metrics across epochs.
This data is saved to json files after each epoch. 
'''

def init_listener(no_metrics = False, path = None):
    # set listeners
    if no_metrics:
        exp_listener = Empty_listener()
    else:
        exp_listener = Listener()

    exp_listener.add_meters('train', make_meters(empty=no_metrics))


    return exp_listener

class Empty_listener(object):

    def __init__(self):
        """ Create an empty Listener
        """
        super(Empty_listener, self).__init__()

        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)
    
    def add_meters(self,tag,meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)
            
    def add_meter(self, tag, name, meter):
        assert name not in list(self.meters[tag].keys()), \
            "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter
    
    def log_meters(self,*args,n=None):
        pass
    
    def last_metrics(self, tag = "train"):
        return {}

    def reset_meters(self,tag):
        return self.get_meters(tag)
    
    def get_meters(self, tag):
        assert tag in list(self.meters.keys())
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys())
        assert name in list(self.meters[tag].keys())
        return self.meters[tag][name]
        
    def save(self,*args,safe=False):
        pass
    
    def load(self,*args):
        pass
    
    def to_json(self,*args):
        pass

    def from_json(self,*args):
        pass

class Listener(object):

    def __init__(self):
        """ Create a Listener
        """
        super(Listener, self).__init__()

        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')

        self.info = defaultdict(dict)
        self.logged = defaultdict(dict)
        self.meters = defaultdict(dict)

    def add_meters(self, tag, meters_dict):
        assert tag not in (self.meters.keys())
        for name, meter in meters_dict.items():
            self.add_meter(tag, name, meter)

    def add_meter(self, tag, name, meter):
        assert name not in list(self.meters[tag].keys()), \
            "meter with tag {} and name {} already exists".format(tag, name)
        self.meters[tag][name] = meter

    def log_meter(self, tag, name, n=1):
        meter = self.get_meter(tag, name)
        if name not in self.logged[tag]:
            self.logged[tag][name] = {}
        self.logged[tag][name][n] = meter.value()

    def log_meters(self, tag, n=1):
        for name, meter in self.get_meters(tag).items():
            self.log_meter(tag, name, n=n)
        
    def last_metrics(self, tag="train"):
        # returns the last recorded metrics
        return { name : self.logged[tag][name][-1] for name in self.logged[tag] }

    def reset_meters(self, tag):
        meters = self.get_meters(tag)
        for name, meter in meters.items():
            meter.reset()
        return meters

    def get_meters(self, tag):
        assert tag in list(self.meters.keys())
        return self.meters[tag]

    def get_meter(self, tag, name):
        assert tag in list(self.meters.keys())
        assert name in list(self.meters[tag].keys())
        return self.meters[tag][name]

    def save(self,path,safe = False, verbose = False):
        if verbose:
            print("Saving metrics...",end="")
        self.to_json(path)
        if verbose:
            print("done!")

    def load(self,path,verbose = False):
        if verbose:
            print("Loading metrics...",end="")
        self.from_json(path)
        if verbose:
            print("done!")

    def to_json(self, filename):
        var_dict = copy.copy(vars(self))
        var_dict.pop('meters')
        for key in ('viz', 'viz_dict'):
            if key in list(var_dict.keys()):
                var_dict.pop(key)    
        with open(filename, 'w') as f:
            json.dump(var_dict, f)

    def from_json(self, filename):
        with open(filename, 'r') as f:
            var_dict = json.load(f)
        self.date_and_time = var_dict['date_and_time']
        self.logged = var_dict['logged']
        
        if 'info' in var_dict:
            self.info = var_dict['info']
        # self.name = var_dict['name']