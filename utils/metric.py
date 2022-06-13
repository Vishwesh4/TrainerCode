from creationism.registration.factory import RegistrantFactory
from typing import Dict, List,Union
import torchmetrics
import torch
from utils.logger import Logger

class Metric(RegistrantFactory):
    """
    Wrapper around torchmetrics MetricCollection to incorporate
    logging and multiple MetricCollection objects
    """
    SETTINGS = ["train","val","test"]
    def __init__(self,metrics:Union[torchmetrics.MetricCollection,List[torchmetrics.MetricCollection]],logger:Logger,device:torch.device) -> None:
        """
        Wraps list of metric collections to make it perform like
        singular metric collections. Usefull for multioutput cases.
        Please ensure these objects have seperate prefix so that it could be differentiated
        This class will automatically take care of train/val metrics,
        hence do not clone seperately for train/test/val logs
        Parameters:
            metrics: Single or list of metriccollections
            logger: Logger object for logging purposes
        """
        global mode
        self.metrics = metrics
        self.logger = logger

        if not isinstance(self.metrics,list):
            self.metrics = list(self.metrics)

        self.metrics_dict = {}
        for name in Metric.SETTINGS:
            temp_metric = []
            for mets in self.metrics:
                temp_metric.append(mets.clone(prefix=name+"_").to(device))
            self.metrics_dict[name] = temp_metric

    def compute(self):
        """
        Calculate the aggregate for the epoch and reset
        The results are stored in metrics_calc attributes
        """
        self.metrics_calc = []
        for i in range(len(self.metrics_dict[mode])):
            self.metrics_calc.append(self.metrics_dict[mode][i].compute())
        self.reset()
    
    def reset(self):
        for i in range(len(self.metrics_dict[mode])):
            self.metrics_dict[mode][i].reset()
    
    def __call__(self, *args ):
        """
        For multiple outputs, the argument should be passed as a
        list of tuples, for instance metric([(tensor,tensor),(tensor,tenosr)])
        For single output, the arguments can be two tensors,
        for instance metric(tensor,tensor)
        """
        if not isinstance(args,list):
            args = list(args)
        
        assert len(self.metrics_dict[mode])==len(args),"Number of arguments doesn't match , number of MetricCollection objects provided"
        
        for i,argument in args:
            self.metrics_dict[mode][i](*argument)
    
    def results(self) -> Dict:
        metric_results = {}
        for results in self.metrics_calc:
            diff_results = results.keys()
            for key in diff_results:
                metric_results[key] = results[key].cpu().numpy()
        return metric_results
    
    def log(self):
        results = self.results()
        self.logger.log(results)
        
