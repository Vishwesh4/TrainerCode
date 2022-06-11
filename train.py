import torch, os
import torch._utils
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
from pathlib import Path
import time
import torchvision
import torch.nn as nn
import wandb
import albumentations
from albumentations.pytorch import ToTensorV2
from utils.dataloader import TILS_dataset_Bihead_Area
from utils.parallel import DataParallelModel, DataParallelCriterion, gather
from utils.model import Resnet_bihead
import numpy as np
import torchmetrics
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from utils.config import Config

class Trainer(Config):
    def intialize_wandb(self):
        pass
    def save_checkpoint(self):
        pass
    def load_checkpoint(self):
        pass
    def run_engine(self,*args):
        for i in range(100):
            log_variables,outputs = train_step
            metrics = self.metric_module(inputs,outputs)
            wandb.log(metrics)

    def run(self):
        for epoch in range(epoch_start,data_hyp['EPOCHS']):
            print("EPOCH: {}".format(epoch))
            self.run_engine(train_step)
            self.run_engine(valid_step)

            ### Saving model checkpoints
            if is_save and (epoch+1) % 6 == 0:
                self.save_checkpoint()
                best_metric = metric
            print(f"Metric at save: {best_metric}")
            self.scheduler.step(test_loss)
            if self.is_wandb:
                wandb.log({"Learning rate":optimizer.param_groups[0]["lr"]})
        print("Training done...")

