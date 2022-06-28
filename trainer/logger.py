from typing import Dict, List, Union, Optional

import wandb

from . import RegistrantFactory


class Logger(RegistrantFactory):
    """
    A class used to log variables to wandb primarily. It additionally
    can track variables.
    This class is child of RegistrantFactory and can have subclasses incase user wants to specify special functions
    for their needs
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
        self.kwargs = kwargs
        wandb.run.name = run_name
        wandb.run.save()
        self.tracked_vars = {}

    def watch(self, **kwargs) -> None:
        """
        Method with arguments similar to wandb.watch
        Please refer to its documentation
        """
        self.logging.watch(**kwargs)

    def log(self, var: dict, **kwargs) -> None:
        """
        Method with arguments similar to wandb.log
        Parameters:
            var (Dict): Of form {"PLOT NAME": VARIABLE VALUE}
        Please refer to its documentation
        """
        self.logging.log(var, **kwargs)

    def track(self, **kwargs) -> None:
        """
        Tracks variables based on the name and value provided
        Provide input as logger.track(loss=lossval,metric=metricval,...)
        """
        if len(self.tracked_vars) == 0:
            self._initialize_track(kwargs)
        else:
            for key in kwargs.keys():
                self.tracked_vars[key].append(kwargs[key])

    def _initialize_track(self, dicts: Dict) -> None:
        for key in dicts.keys():
            self.tracked_vars[key] = [dicts[key]]

    def get_tracked(self, key: str) -> List:
        """
        Gets the desired variables as dictinary, given keys
        """
        return self.tracked_vars[key]

    def reset(self):
        self.tracked_vars = {}

    @property
    def get_logger(self) -> wandb.run:
        """
        Gets the logger object
        """
        return self.logging
