import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time

from args import parse_args
from toolbox.experiment import Experiment
from toolbox.image_preprocessing import image_loader
from models.base_models import *



class ContentLoss(nn.Module):

    def __init__(self, target, weight = 1):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = -1
        self.weight = weight

    def forward(self, input):
        self.loss = self.weight * F.mse_loss(input, self.target)
        return input


class Calculator():

    def __init__(self, device="cpu"):

        self.device = device
        self.imsize = (512,512) if device == "cuda" else (128,128)
        self.content_layers = ['conv4_2']

        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

    def setup_reference(self, reference):
        content_losses = []

        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)
        self.model = nn.Sequential(normalization)

        self.content_image = image_loader(reference, self.imsize).to(self.device, torch.float)[:,:3,:,:]

        num_pool, num_conv = 0, 0

        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                num_conv += 1
                name = "conv{}_{}".format(num_pool, num_conv)

            elif isinstance(layer, nn.ReLU):
                name = "relu{}_{}".format(num_pool, num_conv)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                num_pool += 1
                num_conv = 0
                name = "pool_{}".format(num_pool)
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )

            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn{}_{}".format(num_pool, num_conv)

            elif isinstance(layer, Normalization):
                name = "normalization"

            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            self.model.add_module(name, layer)

            if name in self.content_layers:
                # if we are resuming, the loss layers are already created
                target = self.model(self.content_image).detach()
                content_loss = ContentLoss(target, weight=1 / len(self.content_layers))
                self.model.add_module("content_loss_{}".format(num_pool), content_loss)
                content_losses.append(content_loss)

        self.content_losses = content_losses


    def run(self, target, *args):

        self.input_image = image_loader(target, self.imsize).to(self.device, torch.float)

        self.input_image.data.clamp_(0, 1)

        self.model(self.input_image[:,:3,:,:])

        content_loss = sum(map(lambda x: 10e2 * x.loss, self.content_losses))

        return content_loss

    def distance_matrix(self,list):
        matrix = np.zeros((len(list),len(list)))
        for i in range(len(list)):
            self.setup_reference(list[i])
            for j in range(i+1,len(list)):
                matrix[i,j] = self.run(list[j])
        matrix = matrix + matrix.T
        return matrix




def main():
    args = parse_args()

    list = ["ci.jpg", "ci2.jpg", "ci3.jpg","ci4.jpg"]

    calc = Calculator(args)
    print(calc.distance_matrix(list))
