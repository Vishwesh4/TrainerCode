import abc
from creationism.registration.factory import RegistrantFactory

import torch
import torch._utils
# import torch.nn.functional as F
import torch.nn as nn
import torchvision
import warnings

class Model(RegistrantFactory,nn.Module):
    def load_model_weights(self, model_path,device):
        """
        Loads model weight
        """
        state = torch.load(model_path, map_location=device)
        state_dict = state['model_state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('resnet.', '').replace('module.', '')] = state_dict.pop(key)
        model_dict = self.state_dict()   
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys())!=len(model_dict.keys()):
            warnings.warn("Warning... Some Weights could not be loaded")
        if weights == {}:
            warnings.warn("Warning... No weight could be loaded..")
        model_dict.update(weights)
        self.load_state_dict(model_dict)
    
    def dataparallel(self,device_list):    
        """
        Converts the model to dataparallel to use multiple gpu
        """
        self = torch.nn.DataParallel(self,device_ids=device_list)
        
@Model.register(("resnetbihead",))
class Resnet_bihead(Model):
    def __init__(self, model_name,pretrained=False):
        super(Resnet_bihead, self).__init__()
        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        
        #Cell Head
        self.cell = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                  nn.Linear(400,200),
                                  nn.Linear(200,1),
                                  torch.nn.Sigmoid())

        #Tissue Head
        self.tissue = nn.Sequential(nn.Linear(self.model.fc.out_features,400),
                                    nn.Linear(400,200),
                                    nn.Linear(200,1),
                                    torch.nn.Sigmoid())

    def forward(self, image,head="all"):
        feat = self.model(image)
        if head=="all":
            cell_score = self.cell(feat)
            tissue_score = self.tissue(feat)
            return cell_score,tissue_score
        elif head=="cell":
            return self.cell(feat)
        elif head=="tissue":
            return self.tissue(feat)
        else:
            raise ValueError("Incorrect head given, choose out of all/cell/tissue")
