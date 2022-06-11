"""
This script is a wrapper around the data loader. The class defined returns data loaders
"""
import abc
from creationism.registration.factory import RegistrantFactory
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from .dataloader import TILS_dataset_Bihead_Area

class Dataset(RegistrantFactory):
    """
    Main Dataset class which 
    """
    def __init__(
        self,
        path:str,
        train_batch_size:int,
        test_batch_size:int,
        transform_train,
        transform_test
    ) -> None:
        
        
        self.path = path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.transform_train = transform_train
        self.transform_test = transform_test
        
        #Empty Variables
        self.train_loader = None
        self.test_loader = None
        self.trainset = None
        self.testset = None

    @abc.abstractmethod
    def get_loaders(self)->None:
        """
        This method is supposed to overwrite and get the train and test dataset and dataloaders ready
        """
    @property
    def trainloader(self):
        return self.train_loader
    @property
    def testloader(self):
        return self.test_loader
    @property
    def trainset(self):
        return self.trainset
    @property
    def testset(self):
        return self.testset


@Dataset.register(("tils",))
class TILSDataset(Dataset):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
    
    def _process_data(self):
        df1 = pd.read_csv("/localdisk3/ramanav/TIL_Patches_v2/TILcount.csv")
        df2 = pd.read_csv("/localdisk3/ramanav/TIL_Patches_v2/3_tilcount.csv")
        df3 = pd.read_csv("/localdisk3/ramanav/TIL_Patches_v2/4_tilcount.csv")
        df1 = df1.sample(frac=0.5)
        til = pd.concat([df1,df2,df3]).reset_index(drop=True)
        # til = pd.read_csv(til_file)
        # til["TILdensity"] = (til["TILS_1"]+til["TILS_2"]+til["TILS_3"])/(til["class_2"]+til["class_3"]+til["class_7"]+0.000001)
        til["Rest"] = til["class_1"]+til["class_4"]+til["class_5"]+til["class_6"]+til["class_8"]
        til["TILarea"] = til["TILS_1"] + til["TILS_2"] + til["TILS_3"]
        til["total_area"] = til["Rest"] + til["class_2"] + til["class_3"] + til["class_7"]
        til["metric"] = 1.2*(til["TILarea"]/til["total_area"])+1.2*(til["class_7"]/til["total_area"])+0.4*(((til["class_2"]+til["class_3"])/til["total_area"]))
        til["sample"] = 0
        til.loc[til["metric"]>0,"sample"] = 1
        til.loc[til["metric"]>0.4,"sample"] = 2
        til.loc[til["metric"]>0.7,"sample"] = 3
        til["weight"] = 0
        til.loc[til["sample"]==0,"weight"] = 1/(til["sample"]==0).sum()
        til.loc[til["sample"]==1,"weight"] = 1.5/(til["sample"]==1).sum()
        til.loc[til["sample"]==2,"weight"] = 1.5/(til["sample"]==2).sum()
        til.loc[til["sample"]==3,"weight"] = 1/(til["sample"]==3).sum()

        X = np.arange(0,len(til))
        Y = til["sample"].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42,stratify=Y)
        train_dataset = til.iloc[X_train].copy().reset_index(drop=True)
        test_dataset = til.iloc[X_test].copy().reset_index(drop=True)
        return train_dataset,test_dataset

    def get_loaders(self):
        train_dataset,test_dataset = self._process_data()
        #Weighted sampling
        sample_weights = torch.from_numpy(train_dataset["weight"].values)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))

        #Loading the images for train set
        self.trainset = TILS_dataset_Bihead_Area(data_file=train_dataset,labels=[0,1,5,6],labels_tils=[1,2],path=self.path,transform=self.transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.train_batch_size, num_workers=8,sampler=sampler,pin_memory=True)
        #Loading the images for test set
        self.testset = TILS_dataset_Bihead_Area(data_file=test_dataset,labels=[0,1,5,6],labels_tils=[1,2],path=self.path,transform=self.transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.test_batch_size,shuffle=True, num_workers=8,pin_memory=True)

###################################
#          Data Loaders           #
###################################  
# til_file = dataset_path / Path("TILcount_v2.csv")

