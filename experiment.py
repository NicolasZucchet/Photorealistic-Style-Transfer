import copy
import time
import torch
import json
import torch.nn as nn
import torchvision.models as models
from toolbox.optimizers import set_optimizer_and_scheduler
from closed_form_matting import compute_laplacian
from toolbox.losses import StyleLoss,AugmentedStyleLoss,ContentLoss
from toolbox.utils import Normalization
from toolbox.image_preprocessing import image_loader, masks_loader, tensor_to_image, image_to_tensor
from toolbox.metrics_listener import init_listener
import logging
import matplotlib.pyplot as plt

class Experiment():

    def __init__(self,args):

        self.parameters = Experiment_parameters(args)
        if args.resume_model:
            self.parameters.load() 

        logging.basicConfig(filename=args.res_dir+"experiment.log",format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
        self.log = logging.getLogger("main")

        self.listener = init_listener(self.parameters.resume_model,self.parameters.load_path+"listener.json",self.parameters.no_metrics,self.parameters.verbose)
        if args.resume_model:
            self.listener.load(args.save_listener_path)

        self.style_image = image_loader(args.style_image_path, self.parameters.imsize).to(args.device, torch.float)
        self.content_image = image_loader(args.content_image_path, self.parameters.imsize).to(args.device, torch.float)
        self.input_image = self.content_image.clone()
        self.log.info("images loaded")

        self.style_masks, self.content_masks = masks_loader( args.seg_style_path, args.seg_content_path, self.parameters.imsize)
        for i in range(len(self.style_masks)):
            self.style_masks[i] = self.style_masks[i].to(args.device)
            self.content_masks[i] = self.content_masks[i].to(args.device)
        self.log.info("masks loaded")


        if self.parameters.reg:
            self.L = compute_laplacian(tensor_to_image(self.content_image))
            self.log.info("laplacian computed")

        if args.resume_model:
            self.load()
            # loads the epoch  (and current loss values in the future!!)
        
        else:
            self.epoch = 0

        set_optimizer_and_scheduler(self)
        self.log.info("optimizer and scheduler created")

        self.construct_model()
        self.log.info("model constructed")

        self.log.info("Finished initialising Experiment")

    def disp(self):
        print('----- Experiments Parameters -----')
        for k, v in self.parameters.__dict__.items():
            if k in []:
                continue
            print(k, ':', v)

    def regularization_grad(self):
        """
        Photorealistic regularization
        See Luan et al. for the details.
        """
        im = tensor_to_image(self.input_image)
        grad = self.L.dot(im.reshape(-1, 3))
        loss = (grad * im.reshape(-1, 3)).sum()
        new_grad = 2. * grad.reshape(*im.shape)
        return loss, new_grad
    
    def construct_model(self):
        """
        Assumptions:
            - cnn is a nn.Sequential
            - resize happens only in the pooling layers
        """

        if self.parameters.base_model == "quick":
            content_losses = []
            style_losses = []
            style_masks = copy.deepcopy(self.style_masks)
            content_masks = copy.deepcopy(self.content_masks)

            normalization_mean = torch.tensor([1,1,1]).to(self.parameters.device,torch.float)
            normalization_std = torch.tensor([1,1,1]).to(self.parameters.device,torch.float)
            normalization = Normalization(normalization_mean, normalization_std).to(self.parameters.device)
            model = nn.Sequential(normalization)

            model.add_module(self.parameters.content_layers[0], nn.Conv2d(3,3,5).to(self.parameters.device))
            target = model(self.content_image).detach()
            content_loss = ContentLoss(target).to(self.parameters.device)
            model.add_module("content_loss_1", content_loss)
            content_losses.append(content_loss)

            model.add_module(self.parameters.style_layers[0], nn.Conv2d(3,3,5).to(self.parameters.device))
            target_feature = model(self.style_image).detach()
            style_masks = [model(mask) for mask in style_masks]
            content_masks = [model(mask) for mask in content_masks]
            style_loss = AugmentedStyleLoss(target_feature, style_masks, content_masks).to(self.parameters.device)
            model.add_module("style_loss_1", style_loss)
            style_losses.append(style_loss)

            self.model = model
            self.style_losses = style_losses
            self.content_losses = content_losses
            return

        elif self.parameters.resume_model:
            # we load the saved model as a base model
            cnn = torch.load(self.parameters.load_model_path)
            model = nn.Sequential()

        elif self.parameters.base_model == "vgg19":
            cnn = models.vgg19(pretrained=True).features.to(self.parameters.device).eval()
            normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.parameters.device)
            normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.parameters.device)
            normalization = Normalization(normalization_mean, normalization_std).to(self.parameters.device)
            model = nn.Sequential(normalization)


        else:
            raise Exception("Unrecognized base model requested :"+str(self.parameters.base_model))

        self.log.info("base model {} loaded".format(self.parameters.base_model))

        #cnn = copy.deepcopy(cnn)

        # copying layers form the base model (cnn) and adding the loss layers

        content_losses = []
        style_losses = []
        style_masks = copy.deepcopy(self.style_masks)
        content_masks = copy.deepcopy(self.content_masks)


        num_pool, num_conv = 0, 0
        if self.parameters.resume_model:
            num_cl, num_sl = 0,0
        for layer in cnn.children():
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

                # Update the segmentation masks to match
                # the activation matrices of the neural responses.
                style_masks = [layer(mask) for mask in style_masks]
                content_masks = [layer(mask) for mask in content_masks]

            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn{}_{}".format(num_pool, num_conv)
            
            elif isinstance(layer,Normalization):
                name = "normalization"

            elif isinstance(layer,ContentLoss) and self.parameters.resume_model:
                name = "content_loss{}".format(num_cl)
                num_cl += 1
                content_losses.append(layer)

            elif isinstance(layer,StyleLoss) and self.parameters.resume_model:
                name = "style_loss{}".format(num_sl)
                num_sl += 1
                style_losses.append(layer)

            elif isinstance(layer,AugmentedStyleLoss) and self.parameters.resume_model:
                name = "style_loss{}_augmented".format(num_sl)
                num_sl += 1
                style_losses.append(layer)

            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)


            if not(self.parameters.resume_model) and name in self.parameters.content_layers:
                # if we are resuming, the loss layers are already created
                target = model(self.content_image).detach()
                content_loss = ContentLoss(target, weight= 1/self.parameters.content_layers)
                model.add_module("content_loss_{}".format(num_pool), content_loss)
                content_losses.append(content_loss)

            if not(self.parameters.resume_model) and name in self.parameters.style_layers:
                # if we are resuming, the loss layers are already created
                target_feature = model(self.style_image).detach()
                style_loss = AugmentedStyleLoss(target_feature, style_masks, content_masks, weight= 1/self.parameters.content_layers)
                model.add_module("style_loss_{}".format(num_pool), style_loss)
                style_losses.append(style_loss)
            


        # Trim off the layers after the last content and style losses
        # to speed up forward pass.
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss, AugmentedStyleLoss)):
                break
        
        self.model = model[: (i + 1)]
        self.style_losses = style_losses
        self.content_losses = content_losses
    
    def run(self):
        """
        Run the style transfer.
        `reg_weight` is the photorealistic regularization hyperparameter 
        """
        self.log.info("Starting the transfering over {} epochs".format(self.parameters.num_epochs))
        self.local_epoch = 0
        while self.local_epoch <= self.parameters.num_epochs:
            self.optimizer.step(self.closure) 

        print() # to cancel the end = "" in closure
        
        self.log.info("Experiment finised")

        self.input_image = self.input_image.data.clamp_(0, 1)        
    
    def closure(self):
        """
        https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
        """
        start_time = time.time()

        self.input_image.data.clamp_(0, 1)
        self.optimizer.zero_grad()
        self.model(self.input_image)
        meters = self.listener.reset_meters("train")

        style_loss = sum(map(lambda x: x.loss, self.style_losses))
        style_score = self.parameters.style_weight * style_loss
        meters["style_score"].update(style_score.item())

        content_loss = sum(map(lambda x: x.loss, self.content_losses))
        content_score = self.parameters.content_weight * content_loss
        meters["content_score"].update(content_score.item())

        loss = style_score + content_score
        loss.backward()

        # Add photorealistic regularization
        if self.parameters.reg:
            reg_loss, reg_grad = self.regularization_grad()
            reg_grad_tensor = image_to_tensor(reg_grad,device=self.parameters.device)
            self.input_image.grad += self.parameters.reg_weight * reg_grad_tensor
            reg_score = self.parameters.reg_weight * reg_loss
            meters["reg_score"].update(reg_score.item())
            loss += reg_score

        if self.parameters.verbose:
                print(
                "\r epoch {:>4d}:".format(self.epoch),
                "S: {:.3f} C: {:.3f} R: {:.3f}".format(
                    style_score.item(), content_score.item(), reg_score.item() if self.parameters.reg else 0
                    ),
                end = "")

        self.scheduler.step()
        self.local_epoch += 1
        self.epoch += 1

        meters["epoch_time"].update(time.time()-start_time)        
        self.listener.log_meters("train",self.epoch)

        
        return loss    


    def save(self):
        if not(self.parameters.save_model):
            return

        # saving experiment
        with open(self.parameters.save_experiment_path,"w") as f:
            f.write(str(self.epoch))

        # saving exp parameters
        self.parameters.save()

        # saving the model
        torch.save(self.model, self.parameters.save_model_path)
        # WE SHOULD STILL UPDATE THE VALUES OF THE LOSSES WITH THEIR SAVED ONES
            
        # saving the listener
        self.listener.save(self.parameters.save_listener_path)
    
    def load(self):
        if not(self.parameters.resume_model):
            return
        with open(self.parameters.load_experiment_path,"r") as f:
            self.epoch = int(f.readline())
            print(self.epoch)


class Experiment_parameters():

    def __init__(self,args):
        for k,v in args.__dict__.items():
            self.__setattr__(k,v)
        self.imsize = (512, 512)
    
    def save(self):
        with open(self.save_parameters_path,"w") as f:
            json.dump(self.__dict__,f)
    
    def load(self):
        with open(self.load_parameters_path,"r") as f:
            d = json.load(f)
        for k,v in d.items():
            if k in ["resume","keep_params"] or ("path" in k and (not("seg" in k) or not("image" in k))):
                # the parameters we never want to recover
                continue
            if self.keep_params: 
                # if kee_params, all others are loaded
                self.__setattr__(k,v)
            elif "image" in k or "seg" in k or k in ["base_model","scheduler","no_metrics","reg","seed","imsize"]:
                # the parameters we always want to recover
                self.__setattr__(k,v)

