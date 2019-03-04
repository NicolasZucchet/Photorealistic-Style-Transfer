'''
Custom optimizers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7

def get_optim_parameters(model):
    for param in model.parameters():
        yield param

def set_optimizer_and_scheduler(exp):

    if 'sgd' == exp.parameters.optimizer:
        optimizer = torch.optim.SGD([exp.input_image.requires_grad_()],
                            lr=exp.parameters.lr,
                            momentum=exp.parameters.momentum,
                            weight_decay=exp.parameters.weight_decay
                            )
    elif 'adam' == exp.parameters.optimizer:
        optimizer = torch.optim.Adam([exp.input_image.requires_grad_()],
                            lr=exp.parameters.lr,
                            amsgrad=False)
    elif 'lbfgs' == exp.parameters.optimizer:
        optimizer = torch.optim.LBFGS([exp.input_image.requires_grad_()],
                            lr=exp.parameters.lr)
    
    else:
        raise 'Optimizer {} not available'.format(exp.parameters.optimizer)

    exp.optimizer = optimizer
    

    if 'step' == exp.parameters.scheduler:
        exp.log.info(f' --- Setting lr scheduler to StepLR ---')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=exp.parameters.lr_step, gamma=exp.parameters.lr_decay)
        exp.scheduler = easyScheduling(scheduler, lambda : scheduler.step())
    elif 'exponential' == exp.parameters.scheduler:
        exp.log.info(f' --- Setting lr scheduler to ExponentialLR ---')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp.parameters.lr_decay)    
        exp.scheduler = easyScheduling(scheduler, lambda : scheduler.step())
    elif 'plateau' == exp.parameters.scheduler:
        exp.log.info(f' --- Setting lr scheduler to ReduceLROnPlateau ---') 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=exp.parameters.lr_decay, patience=exp.parameters.lr_step)
        exp.scheduler = easyScheduling(scheduler, lambda : scheduler.step(sum(map(lambda x: x.loss, exp.style_losses))))

    else:
        raise f'Scheduler {exp.parameters.scheduler} not available'


class easyScheduling():
    # this class is made only because not all schedulers have the same step function (Plateau requires a "metrics" argument)
    # we don't want the specificiy to appear in our code. We thus encapsulate these schedulers by defining a default step function inside this class

    def __init__(self,scheduler, step_function):
        self.step = step_function
        self.scheduler = scheduler


def adjust_learning_rate(exp, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every exp.step epochs"""
    lr = exp.parameters.lr * (0.1 ** (epoch // exp.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


