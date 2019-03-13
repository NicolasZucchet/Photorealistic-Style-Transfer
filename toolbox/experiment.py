import copy
import time
import torch
import torch.nn as nn
from toolbox.image_preprocessing import image_loader, masks_loader, tensor_to_image, image_to_tensor
import logging


class Experiment():

    def __init__(self,parameters):
	
        log = logging.getLogger("main")
		# images
        self.style_image = image_loader(parameters.style_image_path, parameters.imsize).to(parameters.device, torch.float)
        self.content_image = image_loader(parameters.content_image_path, parameters.imsize).to(parameters.device, torch.float)
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

		# masks
        self.style_masks, self.content_masks = masks_loader(parameters.seg_style_path, parameters.seg_content_path, parameters.imsize)
        for i in range(len(self.style_masks)):
            self.style_masks[i] = self.style_masks[i].to(parameters.device)
            self.content_masks[i] = self.content_masks[i].to(parameters.device)
        log.info("masks loaded")

		# loading/initialising the epoch counter
        if parameters.resume_model:
            self.load()
            # loads the epoch  (and current loss values in the future!!)
        else:
            self.epoch = 0

        self.local_epoch = 0
        self.log = logging.getLogger("main")

        log.info("Finished initialising Experiment")
    
    def save(self,path):
        with open(path,"w") as f:
            f.write(str(self.epoch))
        
    
    def load(self, path):
        with open(path,"r") as f:
            self.epoch = int(f.readline())
            print(self.epoch)
