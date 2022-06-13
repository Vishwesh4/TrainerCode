import abc
from typing import List
from creationism.registration.factory import RegistrantFactory

import torch
import torch._utils
# import torch.nn.functional as F
import torch.nn as nn
import torchvision
import warnings

class Model(RegistrantFactory,nn.Module):
    """
    Abstract class for torch models. Defines two methods for
    loading model weights, converting model to dataparallel
    """
    def load_model_weights(self, model_path:str,device:torch.device):
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
    
    def dataparallel(self,device_list:List[int]):    
        """
        Converts the model to dataparallel to use multiple gpu
        """
        self = torch.nn.DataParallel(self,device_ids=device_list)
