import copy
import time
import torch
import torch.nn as nn
from toolbox.image_preprocessing import image_loader, masks_loader, tensor_to_image, image_to_tensor
from toolbox.segmentation import get_segmentation, merge_mask
import logging


class Experiment():

    def __init__(self,parameters):
    
        log = logging.getLogger("main")
        # images
        self.style_image = image_loader(parameters.style_image_path, parameters.imsize).to(parameters.device, torch.float)
        self.content_image = image_loader(parameters.content_image_path, parameters.imsize).to(parameters.device, torch.float)
        style_mask_origin, height_, width_ = get_segmentation(parameters.seg_style_path, parameters.imsize)
        content_mask_origin, height2, width2 = get_segmentation(parameters.seg_content_path, parameters.imsize)
        self.style_mask,self.content_mask = merge_mask(style_mask_origin,
                                                              content_mask_origin,
                                                              height_,width_,
                                                              height2,width2,
                                                              parameters.device)
        log.info("masks loaded")


        if parameters.input_image == "content":
            self.input_image = self.content_image.clone()
        elif parameters.input_image == "style":
            self.input_image = self.content_image.clone()
        elif parameters.input_image == "white":
            self.input_image = self.content_image.clone()
            self.input_image.fill_(1)
        elif parameters.input_image == "noise":
            self.input_image = self.content_image.clone()
            self.input_image.random_(0,1000).div_(1000)
            log.info("images loaded")


        self.local_epoch = 0
        self.epoch = 0
        self.log = logging.getLogger("main")

        log.info("Finished initialising Experiment")
    
    def save(self,path):
        with open(path,"w") as f:
            f.write(str(self.epoch))
        
    
    def load(self, path):
        with open(path,"r") as f:
            self.epoch = int(f.readline())
            print(self.epoch)
