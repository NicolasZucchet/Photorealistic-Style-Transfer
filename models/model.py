import copy
import time
from torch import nn
from losses import * # .losses ?


class NeuralStyle(nn.Module):

    def __init__(self,model):
        super(NeuralStyle, self).__init__()
        self.model = model
    
    def forward(self, input_image):
        self.model.forward(input_image)

    # save / load
