import torch
from .dataset import Dataset
from .model import Resnet_bihead
from .wand_module import wandb_func
from .logger import metric_module
from .engine import train_step, valid_step

class Config:
    ##[DEFAULT]
    project_name = 1
    save_checkpoints = 1
    save_directory = 1
    save_wandb = 1
    use_dataparallel = 1
    gpu_devices = 1
    location_mod = 1
    random_seed = 1
    resume_checkpoint = 1

    ##[WANDB]
    wandb_project = 1
    wandb_module = wandb_func

    ##[MODEL]
    encoder_name = 1
    model_module = Resnet_bihead
    transfer_weights = 1
    
    ##[DATASET]
    transform_train = 1
    transform_test = 1
    dataset_path = 1
    train_batch_size = 1
    test_batch_size = 1 
    dataloader_module = Dataset(dataset_path,train_batch_size,test_batch_size,transform_train,transform_test)

    ##[MISCELL]
    learning_rate = 1
    lamda = 1
    patience = 1
    loss_module = torch.nn.L1Loss(reduction="sum")
    optimizer_module = torch.optim.Adam(model.parameters(),lr=data_hyp['LEARNINGRATE'], weight_decay = data_hyp['LAMBDA'])
    schedular =  torch.optim.lr_scheduler.ReduceLROnPlateau
    metric_module = metric_module

    ##[RUN_ENGINE]
    n_epochs = 1
    train_step_module = train_step
    valid_step_module = valid_step