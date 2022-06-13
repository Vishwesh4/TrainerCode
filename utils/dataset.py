"""
This script is a wrapper around the data loader. The class defined returns data loaders
"""
import abc
from creationism.registration.factory import RegistrantFactory
from typing import Any, List, Optional, Union


class Dataset(RegistrantFactory):
    """
    Main Dataset class
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
