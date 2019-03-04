import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .utils import gram_matrix

class ContentLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target, weight = 1):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = -1
        self.weight = weight

    def forward(self, input):
        self.loss = self.weight * F.mse_loss(input, self.target)
        return input
            

class StyleLoss(nn.Module):
    """
    See Gatys et al. for the details.
    """

    def __init__(self, target_feature, weight = 1):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach() # detaches from the graph. New object is linked to old one but will never require grad
        self.loss = -1
        self.weight = weight

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = self.weight * F.mse_loss(gram, self.target)
        return input

class AugmentedStyleLoss(nn.Module):
    """
    AugmentedStyleLoss exploits the semantic information of images.
    See Luan et al. for the details.
    """

    def __init__(self, target_feature, target_masks, input_masks, weight = 1):
        super(AugmentedStyleLoss, self).__init__()
        self.input_masks = [mask.detach() for mask in input_masks]
        self.targets = [
            gram_matrix(target_feature * mask).detach() for mask in target_masks
        ]
        self.loss = -1
        self.weight = weight

    def forward(self, input):
        gram_matrices = [
            gram_matrix(input * mask.detach()) for mask in self.input_masks
        ]
        self.loss = self.weight * sum(
            F.mse_loss(gram, target)
            for gram, target in zip(gram_matrices, self.targets)
        )
        return input
