import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from models.closed_form_matting import compute_laplacian
from toolbox.image_preprocessing import tensor_to_image, image_to_tensor
import logging

log = logging.getLogger("main")

class ExperimentLosses():

    def __init__(self, content_weight, style_weight, reg_weight, content_image = None, device = 'cpu'):
        self.content_losses = []
        self.style_losses = []
        self.reg_losses = []

        self.reg_weight = reg_weight
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.current_style_loss = None
        self.current_content_loss = None
        self.current_reg_loss = None

        self.device = device

        if reg_weight > 0 :
            if content_image is None:
                raise Exception("content image should be provided if regularization is demanded")
            laplacian = compute_laplacian(tensor_to_image(content_image))
            values = laplacian.data
            indices = np.vstack((laplacian.row, laplacian.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = laplacian.shape

            self.L = Variable(torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(self.device), requires_grad=False)
            log.info("laplacian computed")


    def add_content_loss(self,loss):
        self.content_losses.append(loss)
    def compute_content_loss(self):
        return sum(map(lambda x: x.loss, self.content_losses)) * self.content_weight
    
    def add_style_loss(self,loss):
        self.style_losses.append(loss)
    def compute_style_loss(self):
        return sum(map(lambda x: x.loss, self.style_losses)) * self.style_weight

    def compute_reg_loss(self,input_image):
        reg_loss, reg_grad = self.regularization_grad(input_image)
        self.current_reg_loss = self.reg_weight * reg_loss
        return self.current_reg_loss
    def regularization_grad(self, image):
        im = tensor_to_image(image)
        img = image.squeeze(0)
        channel, height, width = img.size()
        loss = 0
        grads = list()
        for i in range(channel):
            grad = torch.mm(self.L, img[i, :, :].reshape(-1, 1))
            loss += torch.mm(img[i, :, :].reshape(1, -1), grad)
            grads.append(grad.reshape((height, width)))
        gradient = torch.stack(grads, dim=0).unsqueeze(0)
        return loss, 2. * gradient

    def compute_total_loss(self):
        return self.compute_content_loss.item() + self.compute_reg_loss.item() + self.compute_style_loss.item()


class ContentLoss(nn.Module):
    def __init__(self, target, weight = 1):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = torch.zeros(1)
        self.weight = weight

    def forward(self, input):
        self.loss = self.weight * F.mse_loss(input, self.target)
        return input
            

class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask, content_mask,device):
        super(StyleLoss, self).__init__()

        self.device = device

        self.style_mask = style_mask.detach()
        self.content_mask = content_mask.detach()

        _, channel_f, height, width = target_feature.size()
        channel = self.style_mask.size()[0]

        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).to(self.device)
        mask_ = F.grid_sample(self.style_mask.unsqueeze(0), grid).squeeze(0)
        target_feature_3d = target_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        target_feature_masked = torch.zeros(size_of_mask, dtype=torch.float).to(self.device)
        for i in range(channel):
            target_feature_masked[i, :, :, :] = mask_[i, :, :] * target_feature_3d

        self.targets = list()
        for i in range(channel):
            if torch.mean(mask_[i, :, :]) > 0.0:
                temp = target_feature_masked[i, :, :, :]
                self.targets.append(gram_matrix(temp.unsqueeze(0)).detach() / torch.mean(mask_[i, :, :]))
            else:
                self.targets.append(gram_matrix(temp.unsqueeze(0)).detach())

    def forward(self, input_feature):
        self.loss = 0
        _, channel_f, height, width = input_feature.size()
        # channel = self.content_mask.size()[0]
        channel = len(self.targets)
        # ****
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).to(self.device)
        mask = F.grid_sample(self.content_mask.unsqueeze(0), grid).squeeze(0)
        # ****
        # mask = self.content_mask.data.resize_(channel, height, width).clone()
        input_feature_3d = input_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        input_feature_masked = torch.zeros(size_of_mask, dtype=torch.float32).to(self.device)
        for i in range(channel):
            input_feature_masked[i, :, :, :] = mask[i, :, :] * input_feature_3d

        inputs_G = list()
        for i in range(channel):
            temp = input_feature_masked[i, :, :, :]
            mask_mean = torch.mean(mask[i, :, :])
            if mask_mean > 0.0:
                inputs_G.append(gram_matrix(temp.unsqueeze(0)) / mask_mean)
            else:
                inputs_G.append(gram_matrix(temp.unsqueeze(0)))
        for i in range(channel):
            mask_mean = torch.mean(mask[i, :, :])
            self.loss += F.mse_loss(inputs_G[i], self.targets[i]) * mask_mean

        return input_feature

class AugmentedStyleLoss(nn.Module):
    def __init__(self, target_feature, target_masks, input_masks, weight = 1):
        super(AugmentedStyleLoss, self).__init__()
        self.input_masks = [mask.detach() for mask in input_masks]
        self.targets = [
            gram_matrix(target_feature * mask).detach() for mask in target_masks
        ]
        self.loss = torch.zeros(1)
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


def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    gram = torch.mm(features, features.t())

    return gram.div(B * C * H * W)