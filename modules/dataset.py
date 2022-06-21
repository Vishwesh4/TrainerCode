"""
This script is a wrapper around the data loader. The class defined returns data loaders
"""
import abc
from typing import Any, List, Optional, Union, Tuple

from torch.utils.data import Dataset, DataLoader

from .register import RegistrantFactory


class Dataset(RegistrantFactory):
    """
    Main Dataset class
    """

    def __init__(self, path: str, train_batch_size: int, test_batch_size: int,) -> None:

        self.path = path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Empty Variables
        self.train_loader = None
        self.test_loader = None
        self.trainset = None
        self.testset = None

        self.train_transform, self.test_transform = self.get_transforms()
        self.trainset, self.train_loader, self.testset, self.test_loader = self.get_loaders()


    @abc.abstractmethod
    def get_transforms(self) -> Tuple(Any,Any):
        """
        In this method, you are supposed to define transforms for augmentation purposes
        Returns:
            train transforms
            test transforms
        """

    @abc.abstractmethod
    def get_loaders(self) -> Tuple(Dataset, DataLoader, Dataset, DataLoader):
        """
        This method is supposed to overwrite and get the train and test dataset and dataloaders ready
        Returns:
            trainset
            train loader
            testset
            test loader
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