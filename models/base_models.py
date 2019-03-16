from torch import nn
import torch
import torchvision.models as models

def get_base_model(*args):
    return 

class QuickModel(nn.Module):

    def __init__(self,parameters):
        super(QuickModel, self).__init__()
        model = nn.Sequential()
        normalization_mean = torch.tensor([1,1,1]).to(parameters.device,torch.float)
        normalization_std = torch.tensor([1,1,1]).to(parameters.device,torch.float)
        normalization = Normalization(normalization_mean, normalization_std).to(parameters.device)
        model.add_module("normalization",normalization)
        model.add_module("quick_loss_style", nn.Conv2d(3,3,5).to(parameters.device))
        model.add_module("quick_loss_content", nn.Conv2d(3,3,5).to(parameters.device))
        self.model = model
    
    def children(self):
        return self.model.children()
    
    def forward(self, input_image):
        self.model.forward(input_image)




class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
        
    


