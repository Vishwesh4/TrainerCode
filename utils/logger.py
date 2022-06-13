from creationism.registration.factory import RegistrantFactory
import wandb

class Logger(RegistrantFactory):
    def __init__(self) -> None:
        self.logging = wandb.init(project=project_name,config=config_path,resume=resume,notes=notes)
        self.logging.run.name = run_name
        self.logging.run.save()

    def watch(self,**kwargs):
        self.logging.watch(kwargs)

    def log(self,**kwargs):
        self.logging.log(kwargs)

    def track(self):
        pass

    def compute_mean(self):
        pass

    def reset(self):
        pass
