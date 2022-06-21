"""
Trainer Class, Written for general purpose training. Aim is to have a better code base, learn how to write proper object oriented programming
and while shifting from experiment to experiment, dont have to change main code again and again
Author: Vishwesh Ramanathan
Created: June 3, 2022
"""
import os
import abc
from typing import Union, Dict
import time
import inspect
import random

from pathlib import Path
import torch
import numpy as np
import yaml

from . import Dataset, Model, Metric, Logger


class Trainer:
    # Global variable, mode should be one of either train/val/test
    mode = "train"

    def __init__(self, config_pth:str) -> None:
        with open(config_pth,"r") as file:
            self.args = yaml.safe_load(file)

        #Initializes for training, gets folders ready
        self.initialize_engine()

        self.device = torch.device(f'cuda:{self.args["DEFAULT"]["gpu_devices"][0]}' if torch.cuda.is_available() else "cpu")
        
        # Build classes
        self.dataset = Dataset.create(**self.args["DATASET"])
        self.logger = Logger.create(**self.args["LOGGER"])
        self.model = Model.create(**self.args["MODEL"])
        self.metrics = Metric.create(**self.args["METRIC"])
        
        self.loss_fun, self.optimizer, self.scheduler = self.get_ops(self.model, self.args["LOSS"], self.args["OPTIMIZER"], self.args["SCHEDULER"])
        self.loss_fun = self.loss_fun.to(self.device)
        
    def initialize_engine(self):
        """
        Makes new directory for saving model if applicable and initializes seed
        """
        #Specifically for computecanada
        if self.args["DEFAULT"]["location_mod"] is not None:
            self.args["DATASET"]["path"] = str(Path(self.args["DEFAULT"]["location_mod"])/self.args["DATASET"]["path"])

        if self.args["DEFAULT"]["save_loc"] is not None:
            self.is_save = True
            parent_name = Path(self.args["DEFAULT"]["save_loc"])
            folder_name = parent_name / Path("Results")
            model_save = folder_name / Path(self.args["LOGGER"]["run_name"]) / "saved_models"
        else:
            self.is_save = False

        if not folder_name.is_dir():
            os.mkdir(folder_name)
            os.mkdir(model_save.parent)
            os.mkdir(model_save)
        elif not model_save.parent.is_dir():
            os.mkdir(model_save.parent)
            os.mkdir(model_save)
        else:
            pass
        
        random_seed = self.args["DEFAULT"]["random_seed"]
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def save_checkpoint(self, epoch, metric):
        """
        Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
        """
        print("Saving Checkpoint ...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metric": metric,
            },
            self.args["DEFAULT"]["save_loc"]
            / Path(
                "Checkpoint_{}_{:.2f}.pt".format(
                    time.strftime("%d%b%H_%M_%S", time.gmtime()), metric.item()
                )
            ),
        )
        torch.cuda.empty_cache()

    def load_checkpoint(self):
        """
        Loads checkpoint for the model, optimizer, scheduler, epoch and best_metric
        """
        checkpoint = torch.load(
            self.args["DEFAULT"]["resume_loc"], map_location=f"cuda:{self.args['DEFAULT']['gpu_devices'][0]}"
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        best_metric = checkpoint["metric"]
        del checkpoint
        torch.cuda.empty_cache()
        return epoch, best_metric

    @abc.abstractmethod
    def get_ops(self, loss_args:Dict, opt_args:Dict, schd_args:Dict):
        """
        Method for specifying the loss function, learning rate schedular and
        optimizer
        You can pass like
        loss = LossClass(**loss_args)
        optimizer = OptimizerClass(**opt_args)
        schedular = schedularClass(**schd_args)
        """
        ...

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """
        Method for training loop
        """
        ...

    @abc.abstractmethod
    def val(self, *args, **kwargs) -> Union[float, float]:
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
        if self.args["DEFAULT"]["resume_loc"] is not None:
            epoch_start, best_metric_val = self.load_checkpoint()
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args["DEFAULT"]["epochs"]):
            print("EPOCH: {}, METRIC: {}".format(epoch, best_metric_val))
            # For logging purpose, we need to define the modes
            Trainer.mode = "train"
            self.train()
            Trainer.mode = "val"
            metric_val, val_loss = self.val()
            if self.is_save and (
                ((epoch + 1) % self.args["DEFAULT"]["save_freq"] == 0) or (metric_val > best_metric_val)
            ):
                self.save_checkpoint(epoch, metric_val)
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
            if "metrics" in inspect.getfullargspec(self.scheduler.step).args:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            self.logger.log(
                value=self.optimizer.param_groups[0]["lr"], name="Learning Rate"
            )
        print("Training Done ...")
