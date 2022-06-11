
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from typing import List

class MultiFocalLoss(Module):
    def __init__(self,alpha,gamma,target_importance=0.75) -> None:
        super(MultiFocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alpha)
        self.target_importance = target_importance
    
    def _flatten(self,input):
        """
        flattens the matrix of dimension N,C,H,W to N*H*W,C
        """
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))  # N,H*W,C => N*H*W,C
        return input

    def _loss(self,y_pred,y_true):
        #Implements from paper https://arxiv.org/pdf/1708.02002.pdf, FL = -alpha*(1-p)**gamma * log(p)
        #From https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/3
        assert len(y_pred.shape) == 4 #shape of y_pred should be of the form [n_batch,7,256,256]
        y_pred = self._flatten(y_pred)
        y_true = self._flatten(y_true)
        self.alpha = self.alpha.to(y_true.device)
        alpha_mult = torch.matmul(y_true,self.alpha)
        
        ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha_mult * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

    def forward(self,Y_PRED,Y_TRUE):
        y_pred_target,y_pred_context = Y_PRED
        y_true_target,y_true_context = Y_TRUE
        return self.target_importance*self._loss(y_pred_target,y_true_target) + (1-self.target_importance)*self._loss(y_pred_context,y_true_context)

class DiceScore:
    def __init__(self,labels:List=None):
        """
        Parameters:
            label: Only calculate the dice score for labels listed in the list, if None, calculate dice score for all the classes
        """
        self.labels = labels

    def __call__(self,Y_PRED,Y_TRUE):
        y_pred_target,y_pred_context = Y_PRED
        y_true_target,y_true_context = Y_TRUE

        y_pred_target = F.one_hot(torch.argmax(y_pred_target,dim=1),num_classes=y_pred_target.shape[1]).permute(0,3,1,2)
        if self.labels is None:
            return self._DiceScore(y_pred_target,y_true_target)
        else:
            return self._DiceScore(y_pred_target[:,self.labels,:,:],y_true_target[:,self.labels,:,:])

    def _DiceScore(self,y_pred,y_true):
        smooth = 1

        iflat = torch.flatten(y_pred)
        tflat = torch.flatten(y_true)
        intersection = (iflat * tflat).sum()
        
        return ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class MultiFocalLoss_unet(Module):
    def __init__(self,alpha,gamma) -> None:
        super(MultiFocalLoss_unet,self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor(alpha)
    
    def _flatten(self,input):
        """
        flattens the matrix of dimension N,C,H,W to N*H*W,C
        """
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))  # N,H*W,C => N*H*W,C
        return input

    def _loss(self,y_pred,y_true):
        #Implements from paper https://arxiv.org/pdf/1708.02002.pdf, FL = -alpha*(1-p)**gamma * log(p)
        #From https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/3
        assert len(y_pred.shape) == 4 #shape of y_pred should be of the form [n_batch,7,256,256]
        y_pred = self._flatten(y_pred)
        y_true = self._flatten(y_true)
        self.alpha = self.alpha.to(y_true.device)
        alpha_mult = torch.matmul(y_true,self.alpha)
        
        ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha_mult * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

    def forward(self,y_pred_target,y_true_target):
        return self._loss(y_pred_target,y_true_target)

class DiceScore_unet:
    def __init__(self,labels:List=None):
        """
        Parameters:
            label: Only calculate the dice score for labels listed in the list, if None, calculate dice score for all the classes
        """
        self.labels = labels

    def __call__(self,y_pred_target,y_true_target):
        y_pred_target = F.one_hot(torch.argmax(y_pred_target,dim=1),num_classes=y_pred_target.shape[1]).permute(0,3,1,2)
        if self.labels is None:
            return self._DiceScore(y_pred_target,y_true_target)
        else:
            return self._DiceScore(y_pred_target[:,self.labels,:,:],y_true_target[:,self.labels,:,:])

    def _DiceScore(self,y_pred,y_true):
        smooth = 1

        iflat = torch.flatten(y_pred)
        tflat = torch.flatten(y_true)
        intersection = (iflat * tflat).sum()
        
        return ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))