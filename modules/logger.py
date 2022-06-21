from typing import Dict, Union, Optional
import wandb

from .register import RegistrantFactory


class Logger(RegistrantFactory):
    """
    A class used to log variables to wandb primarily. It additionally
    can track variables.
    This class is child of RegistrantFactory and can have subclasses
    """

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
        self.logging.run.name = run_name
        self.logging.run.save()
        self.tracked_vars = {}

    def watch(self, **kwargs):
        """
        Method with arguments similar to wandb.watch
        Please refer to its documentation
        """
        self.logging.watch(kwargs)

    def log(self, **kwargs):
        """
        Method with arguments similar to wandb.log
        Please refer to its documentation
        """
        self.logging.log(kwargs)

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

    def get_tracked(self, *args):
        """
        Gets the desired variables as dictinary, given keys
        """
        subset_tracked = {
            key: value for key, value in self.tracked_vars.items() if key in args
        }
        return subset_tracked

    def reset(self):
        self.tracked_vars = {}
