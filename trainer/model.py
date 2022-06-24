import abc
from typing import List
import warnings

import torch
import torch._utils

# import torch.nn.functional as F
import torch.nn as nn

from . import RegistrantFactory


class Model(RegistrantFactory, nn.Module):
    """
    Abstract class for torch models. Defines two methods for
    loading model weights, converting model to dataparallel
    Methods:
        load_model_weights: For loading model weights of the neural network
    """
    subclasses = {}

    def load_model_weights(self, model_path: str, device: torch.device) -> None:
        """
        Loads model weight
        """
        state = torch.load(model_path, map_location=device)
        if isinstance(state,dict):
            for keys in state.keys():
                if "model" in keys:
                    #Assumes the state dict has name model in its name
                    state_dict = state[keys]
                    break
        else:
            state_dict = state
        for key in list(state_dict.keys()):
            state_dict[
                key.replace("resnet.", "").replace("module.", "")
            ] = state_dict.pop(key)
        model_dict = self.state_dict()
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys()) != len(model_dict.keys()):
            warnings.warn("Warning... Some Weights could not be loaded")
        if weights == {}:
            warnings.warn("Warning... No weight could be loaded..")
        model_dict.update(weights)
        self.load_state_dict(model_dict)
