"""
Trainer Class, Written for general purpose training. Aim is to have a better code base, learn how to write proper object oriented programming
and while shifting from experiment to experiment, dont have to change main code again and again
Author: Vishwesh Ramanathan
Created: June 3, 2022

Modifications:
    1.
"""
import abc
from typing import Union
import time
import inspect
from creationism.registration.factory import RegistrantFactory
from pathlib import Path
import torch

class Trainer(RegistrantFactory):
    #Global variable, mode should be one of either train/val/test
    mode = "train"

    def __init__(self) -> None:
        #Objects dealing with important functionalities related to training/validation/testing/logging functionlities
        self.lossfun = lossfun
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.metric = metric

        #Important Attributes
        self.epochs = epochs
        self.save_loc = save_loc
        self.resume_loc = resume_loc
        self.device_list = device_list

        #Training option attributes
        self.is_save = is_save
        self.save_freq = save_freq

    @classmethod
    def build(cls,config):
        return cls

    def save_checkpoint(self,epoch,metric):
        """
        Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
        """
        print("Saving Checkpoint ...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric
            }, self.save_loc/Path("Checkpoint_{}_{:.2f}.pt".format(time.strftime("%d%b%H_%M_%S",time.gmtime()),metric.item()))
        )
        torch.cuda.empty_cache()

    def load_checkpoint(self):
        """
        Loads checkpoint for the model, optimizer, scheduler, epoch and best_metric
        """
        checkpoint = torch.load(self.resume_loc, map_location=f'cuda:{self.device_list[0]}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_metric = checkpoint['metric']
        del checkpoint
        torch.cuda.empty_cache()
        return epoch,best_metric

    @abc.abstractmethod
    def train(self):
        """
        Method for training loop
        """
        ...

    @abc.abstractmethod
    def val(self)->Union[float, float]:
        """
        Method for validation loop.
        Returns:
            metrics: Metric value for the epoch. Please make sure better performance indicates higher metric score
            loss: Loss value, mainly used for step function if valid
        """
        ...

    def run(self):
        """
        Run training and validation for the number of epochs provided
        """    
        best_metric_val = -10000
        if self.resume_loc is not None:
            epoch_start, best_metric_val = self.load_checkpoint()
        else:
            epoch_start = 0
        
        for epoch in range(epoch_start,self.epochs):
            print("EPOCH: {}, METRIC: {}".format(epoch,best_metric_val))
            #For logging purpose, we need to define the modes
            Trainer.mode = "train"
            self.train()
            Trainer.mode = "val"
            metric_val,test_loss = self.val()
            if self.is_save and (((epoch+1) % self.save_freq == 0) or (metric_val>best_metric_val)):
                self.save_checkpoint(epoch,metric_val)
                if metric_val>best_metric_val:
                    best_metric_val = metric_val
            if "metrics" in inspect.getfullargspec(self.scheduler.step).args:
                self.scheduler.step(metric_val)
            else:
                self.scheduler.step()
            self.logger.log(value=self.optimizer.param_groups[0]["lr"], name="Learning Rate")
        print("Training Done ...")


