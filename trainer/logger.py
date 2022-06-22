from typing import Dict, Union, Optional

import wandb

from . import RegistrantFactory


class Logger(RegistrantFactory):
    """
    A class used to log variables to wandb primarily. It additionally
    can track variables.
    This class is child of RegistrantFactory and can have subclasses
    """
    subclasses = {}
    def __init__(
        self,
        project_name: str,
        configs: Union[Dict, str, None],
        notes: str,
        run_name: str,
        **kwargs
    ) -> None:
        """
        Initilizes wandb and tracking variable attribute
        Parameters:
            project_name = Name of main project
            configs = Hyperparameters/settings used
            notes = Notes for the experiment
            run_name = Name of the experiment
            **kwargs 
        """
        self.logging = wandb.init(
            project=project_name, config=configs, notes=notes, **kwargs
        )
        wandb.run.name = run_name
        wandb.run.save()
        self.tracked_vars = {}

    def watch(self, **kwargs):
        """
        Method with arguments similar to wandb.watch
        Please refer to its documentation
        """
        self.logging.watch(**kwargs)

    def log(self, var:dict, **kwargs):
        """
        Method with arguments similar to wandb.log
        Please refer to its documentation
        """
        self.logging.log(var,**kwargs)

    def track(self, **kwargs):
        """
        Tracks variables based on the name and value provided
        Provide input as logger.track(loss=lossval,metric=metricval)
        """
        if len(self.tracked_vars) == 0:
            self._initialize_track(kwargs)
        else:
            for key in kwargs.keys():
                self.tracked_vars[key].append(kwargs[key])

    def _initialize_track(self, dicts: Dict):
        for key in dicts.keys():
            self.tracked_vars[key] = [dicts[key]]

    def get_tracked(self, key:str):
        """
        Gets the desired variables as dictinary, given keys
        """
        return self.tracked_vars[key]

    def reset(self):
        self.tracked_vars = {}
    
    @property
    def get_logger(self):
       """
       Gets the logger object
       """
       return self.logging
