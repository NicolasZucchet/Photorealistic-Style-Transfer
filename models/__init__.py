import logging
import copy

from models.base_models import *
from models.losses import *

import time

def get_model_and_losses(experiment, parameters, content_image):

    """
    Assumptions:
        - resize happens only in the pooling layers
    """ 

    log = logging.getLogger("main")

    if parameters.base_model == "vgg19":
        cnn = models.vgg19(pretrained=True).features.to(parameters.device).eval()

    else:
        raise Exception("Unrecognized base model requested :"+str(parameters.base_model))

    log.info("base model {} loaded".format(parameters.base_model))
        
    model = nn.Sequential()

    # VGG normalization
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(parameters.device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(parameters.device)
    normalization = Normalization(normalization_mean, normalization_std).to(parameters.device)
    model.add_module("normalizarion", normalization)

    losses = ExperimentLosses(
        parameters.content_weight,
        parameters.style_weight,
        parameters.reg_weight,
        content_image=content_image,
        device=parameters.device)

    #print(style_masks[0].shape)
    #exit(0)

    num_pool, num_conv = 0, 0
    n_loss_layers, total_loss_layers = 0, len(parameters.content_layers)+len(parameters.style_layers)

    count = 0
    for layer in cnn.children():
        count += 1

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

        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn{}_{}".format(num_pool, num_conv)
        
        elif isinstance(layer, Normalization):
            name = "normalization"

        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

        model.add_module(name, layer)
    
        # adding the loss layers when needed

        if name in parameters.content_layers:
            n_loss_layers += 1
            target = model(experiment.content_image).detach()
            content_loss = ContentLoss(target, weight= 1/len(parameters.content_layers))
            model.add_module("content_loss_{}".format(num_pool), content_loss)
            losses.add_content_loss(content_loss)

        if name in parameters.style_layers:
            n_loss_layers += 1
            target_feature = model(experiment.style_image).detach()
            style_loss = StyleLoss(target_feature, experiment.style_mask.detach(), experiment.content_mask.detach(), parameters.device)
            model.add_module("style_loss_{}".format(num_pool), style_loss)
            losses.add_style_loss(style_loss)

    # now we trim off the layers after the last content and style losses to speed up forward pass
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, losses
