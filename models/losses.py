import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.backwards_done = False
        self.device = device

        if reg_weight > 0 :
            if content_image is None:
                raise Exception("content image should be provided if regularization is demanded")
            self.L = compute_laplacian(tensor_to_image(content_image))
            log.info("laplacian computed")

    def add_content_loss(self,loss):
        self.content_losses.append(loss)
    
    def add_style_loss(self,loss):
        self.style_losses.append(loss)
    
    def compute_content_loss(self):
        if len(self.content_losses)==0:
            raise Exception("wtf")
        # self.current_content_loss = sum(map(lambda x: x.loss, self.content_losses)) * self.content_weight
        self.backwards_done = False
        # return self.current_content_loss
        return sum(map(lambda x: x.loss, self.content_losses)) * self.content_weight

    def compute_style_loss(self):
        if len(self.style_losses)==0:
            raise Exception("wtf")
        # self.current_style_loss = sum(map(lambda x: x.loss, self.style_losses)) * self.style_weight
        self.backwards_done = False
        # return self.current_style_loss
        return sum(map(lambda x: x.loss, self.style_losses)) * self.style_weight

    def backward(self):
        raise Exception("Method is deprecated")
        if self.backwards_done:
            raise Exception("Backwards propagation has already been computed")
        loss = self.current_content_loss + self.current_style_loss
        loss.backward(retain_graph = True) # new error popped up asking for this....
        self.backwards_done = True


    def regularization_grad(self, input_image):
        """
        Photorealistic regularization
        See Luan et al. for the details.
        """
        im = tensor_to_image(input_image)
        grad = self.L.dot(im.reshape(-1, 3))
        loss = (grad * im.reshape(-1, 3)).sum()
        new_grad = 2. * grad.reshape(*im.shape)
        return loss, new_grad


    def compute_reg_loss(self,input_image):
        reg_loss, reg_grad = self.regularization_grad(input_image)
        reg_grad_tensor = image_to_tensor(reg_grad,device=self.device)
        input_image.grad += self.reg_weight * reg_grad_tensor # DOES THIS UPDATE THE IMAGE GLOBALY ? 
        self.current_reg_loss = self.reg_weight * reg_loss
        return self.current_reg_loss
    
    def total_loss(self):
        return self.current_content_loss.item() + self.current_reg_loss.item() + self.current_style_loss.item()

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

def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    gram = torch.mm(features, features.t())

    return gram.div(B * C * H * W) # division by float