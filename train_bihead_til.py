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
################################################################ Load arguments #################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="hyperparameter location",required=True)
parser.add_argument("-n", help="project name",required=True)
#For compute canada
parser.add_argument("-l",help="Data location modfier",default=None)
parser.add_argument("-m",help="(True/False) To use dataparallel or not",default="True")
parser.add_argument("-r", help="resume location",default=None)
parser.add_argument("-s",help="(True/False)save checkpoints?",default="False")
parser.add_argument("-w",help="(True/False)save wandb?",default="True")
parser.add_argument("-t",help="Path for loading weights for transfer learning",default=None)

args = parser.parse_args()

hyp = args.p
loc_mod = args.l
name = args.n
multi_gpu = eval(args.m)
resume_location = args.r
is_save = eval(args.s)
is_wandb = eval(args.w)
transfer_load = args.t

with open(hyp,"r") as f: 
	data_hyp=json.load(f) 
print(f"Experiment Name: {name}\nHyperparameters:{data_hyp}")
print("CUDA is available:{}".format(torch.cuda.is_available()))

if is_wandb:
    #For visualizations
    wandb.init(project="Cellularity and Tissue area",config=data_hyp)
    wandb.run.name = name
    wandb.run.save()

if loc_mod is None:
    dataset_path = Path(data_hyp["DATASET_PATH"])
else:
    dataset_path = Path(loc_mod)/Path(data_hyp["DATASET_PATH"])

if is_save:
    PARENT_NAME = Path(data_hyp["SAVE_DIR"])
    FOLDER_NAME = PARENT_NAME / Path("Results")
    MODEL_SAVE = FOLDER_NAME / Path(name) / Path("saved_models")

    if not FOLDER_NAME.is_dir():
        os.mkdir(FOLDER_NAME)
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    elif not MODEL_SAVE.parent.is_dir():
        os.mkdir(MODEL_SAVE.parent)
        os.mkdir(MODEL_SAVE)
    else:
        pass

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
#########################################################################################################################################################

def save_checkpoint(epoch,model,optimizer,scheduler,metric):
    """
    Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
    """
    print("Saving Checkpoint ...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric': metric
        }, MODEL_SAVE/Path("Checkpoint_{}.pt".format(time.strftime("%d%b%H_%M_%S",time.gmtime())))
    )
    torch.cuda.empty_cache()

def load_checkpoint(path,model,optimizer,scheduler):
    """
    Loads the saved checkpoint
    """
    checkpoint = torch.load(path, map_location=f'cuda:{DEVICE_LIST[0]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metric = checkpoint['metric']
    del checkpoint
    torch.cuda.empty_cache()

    return model,optimizer,scheduler,epoch,metric

def train(model,trainloader, optimizer, LossFun,MetricFun):
    running_loss_cell = 0.0
    running_loss_tissue = 0.0

    total_dice = 0
    model.train()

    for data in tqdm(trainloader):
        inputs, labels = data
        #Convert to GPU
        inputs,labels[1],labels[2] = inputs.to(device), labels[1].to(device), labels[2].to(device)

        # initialize gradients to zero
        optimizer.zero_grad() 
        # forward pass
        outputs = model(inputs)
        if torch.cuda.device_count() > 1 and multi_gpu:
            cell_outputs,tissue_outputs = gather(outputs,device)
            # tissue_outputs = gather(outputs[1],device)

        #compute Loss with respect to target
        # cell_outputs = outputs[:,0]
        # tissue_outputs = outputs[:,1]
        loss1 = LossFun(cell_outputs.ravel(), labels[1])
        loss2 = LossFun(tissue_outputs.ravel(), labels[2])
        loss = loss1+loss2

        #Metric Calculation
        # total_dice+=MetricFun(outputs, labels)
        metrics_calc = [MetricFun[0](cell_outputs.ravel(),labels[1]),MetricFun[1](tissue_outputs.ravel(),labels[2])]

        # back propagate
        loss.backward()
        # do SGD step i.e., update parameters
        optimizer.step()
        # by default loss is averaged over all elements of batch
        running_loss_cell += loss1.data
        running_loss_tissue += loss2.data
        running_loss = running_loss_cell+running_loss_tissue


        if is_wandb:
            wandb.log({
                "Epoch Train Loss":loss.data,
            })        

    running_loss = running_loss.cpu().numpy()
    metrics_calc = [MetricFun[0].compute(),MetricFun[1].compute()]

    # print(metrics_calc)
    MetricFun[0].reset()
    MetricFun[1].reset()
    R2score = [np.array([metrics_calc[0]["train_cell_R2Score"].cpu().numpy()]),np.array([metrics_calc[1]["train_tissue_R2Score"].cpu().numpy()])]

    if is_wandb:
        wandb.log({
            "Epoch Train Total Cell R2 Score": R2score[0][0],
            "Epoch Train Total Tissue R2 Score": R2score[1][0],
            "Epoch Train Cell Loss":running_loss_cell/len(trainloader),
            "Epoch Train Tissue Loss":running_loss_tissue/len(trainloader)
        })
    return running_loss

def test(model,testloader,LossFun,epoch,MetricFun):
    running_loss_cell = 0.0
    running_loss_tissue = 0.0
    total_dice = 0
    # evaluation mode which takes care of architectural disablings
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            #Convert to GPU
            inputs,labels[1],labels[2] = inputs.to(device), labels[1].to(device), labels[2].to(device)

            outputs = model(inputs)
            if torch.cuda.device_count() > 1 and multi_gpu:
                cell_outputs,tissue_outputs = gather(outputs,device)
            #compute Loss with respect to target
            loss1 = LossFun(cell_outputs.ravel(), labels[1])
            loss2 = LossFun(tissue_outputs.ravel(), labels[2])
            # loss = loss1+loss2
            #Metric Calculation
            # total_dice+=MetricFun(outputs, labels)
            metrics_calc = [MetricFun[0](cell_outputs.ravel(),labels[1]),MetricFun[1](tissue_outputs.ravel(),labels[2])]
            running_loss_cell += loss1.data
            running_loss_tissue += loss2.data
            running_loss = running_loss_cell+running_loss_tissue
    
    metrics_calc = [MetricFun[0].compute(),MetricFun[1].compute()]
    MetricFun[0].reset()
    MetricFun[1].reset()
    R2score = [np.array([metrics_calc[0]["test_cell_R2Score"].cpu().numpy()]),np.array([metrics_calc[1]["test_tissue_R2Score"].cpu().numpy()])]
    # print(metrics_calc)
    running_loss = running_loss.cpu().numpy()
    # print(f"Total R2 score on test data : {metrics_calc['test_R2Score']}")
    # R2score = np.array([metrics_calc["test_R2Score"].cpu().numpy()])
    if is_wandb and epoch%5==0:
        log_wandb_table_stats(inputs,[cell_outputs,tissue_outputs],labels,epoch,MetricFun)
    if is_wandb:
        wandb.log({
            "Test Total Loss":running_loss /  len(testloader),
            "Test Cell Loss":running_loss_cell /  len(testloader),
            "Test Tissue Loss":running_loss_tissue /  len(testloader),
            "Test Cell R2 Score":R2score[0][0],
            "Test Tissue R2 Score":R2score[1][0],
        })

    return (R2score[0]+R2score[1])/2 , running_loss /  len(testloader)

def log_wandb_table_stats(Input,Output,Truth,epoch,MetricFun):
    # W&B: Create a Table to store predictions for each test step
    columns=["id", "image","Masks","Calculated Cellularity area","Real Cellularity area","Calculated Tissue area","Real Tissue area"]
    test_table = wandb.Table(columns=columns)
    # Y_pred_target=torch.argmax(Output,dim=1)
    Y_true_target=torch.argmax(Truth)
    ##[DATASET].ravel(), Truth[1])
    # r2score = metric_calc["test_R2Score"].cpu().numpy()
    # mse = metric_calc["test_MeanSquaredError"].cpu().numpy()
    for i in range(16):
        idx = f"{epoch}_{i}"
        image = wandb.Image(Input[i].permute(1,2,0).cpu().numpy())

        mask = wandb.Image(image, masks={
            "Tissue mask": {"mask_data" : Y_true_target[i], "class_labels" : CLASSES},
            "Cell mask": {"mask_data" : Truth[0][i,-1,:,:].cpu().numpy(), "class_labels" : {0:"None",1:"Lymphocytes and plasma cells"}}
        })
        # metric_calc = MetricFun(Output[i][0], Truth[1][i])
        test_table.add_data(idx, image,mask,Output[0][i][0],Truth[1][i],Output[1][i][0],Truth[2][i])
    wandb.log({"table_key": test_table})
    # MetricFun.reset()

if __name__=="__main__":
    ###################################
    #          Model Setup            #
    ###################################
    NUM_CLASSES = 1
    DEVICE_LIST = data_hyp["DEVICE_LIST"]
    NUM_GPUS = len(DEVICE_LIST)
    # CLASSES = {0:'non-enhancing tumor core',1:'peritumoral edema',2:'GD-enhancing tumor',3:'background'}
    CLASSES = {0:'invasive tumor',1:'tumor-associated stroma',2:'inflamed stroma',3:'rest'}
    device = torch.device(f"cuda:{DEVICE_LIST[0]}" if torch.cuda.is_available() else "cpu")
    

    if (transfer_load is not None) and (resume_location is None):
        print("Loading pretraining weights from previous experiments...")
        model = Resnet_bihead(data_hyp["ENCODER_NAME"],pretrained=False)
        model.load_model_weights(transfer_load,device)
    else:
        model = Resnet_bihead(data_hyp["ENCODER_NAME"],pretrained=True)

    #Multiple GPUs
    if torch.cuda.device_count() > 1  and multi_gpu:
        print("Using {} GPUs".format(NUM_GPUS))
        model = DataParallelModel(model, device_ids=DEVICE_LIST)
    
    model = model.to(device)

    ###################################
    #        Data Augmentations       #
    ###################################   
    
    transform_train =  albumentations.Compose([    
        albumentations.Resize(256,256),
        albumentations.ToFloat(max_value=255),
        albumentations.OneOf([
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),              
            albumentations.Rotate(p=0.5)
            ],p=0.8),
        albumentations.OneOf([
            albumentations.GaussianBlur(p=0.4),
            albumentations.RandomBrightnessContrast(p=0.4),
            # albumentations.Sharpen(p=0.4)
        ],p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=0.9, sat_shift_limit=0.9, val_shift_limit=0.9,p=0.4),
        # albumentations.Affine(scale=(0.8,1.2),translate_percent=(0,0.3),shear=(-10,10),p=0.4),
        ToTensorV2()])
    # transform_train =  albumentations.Compose([
    #     ToTensorV2()]) 
    transform_test =  albumentations.Compose([
        albumentations.Resize(256,256),
        albumentations.ToFloat(max_value=255),
        ToTensorV2()])
    ###################################
    #          Data Loaders           #
    ###################################  
    # til_file = dataset_path / Path("TILcount_v2.csv")
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
    
    #Weighted sampling
    sample_weights = torch.from_numpy(train_dataset["weight"].values)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))
    #Loading the images for train set
    trainset = TILS_dataset_Bihead_Area(data_file=train_dataset,labels=[0,1,5,6],labels_tils=[1,2],path=dataset_path,transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=data_hyp["TRAIN_BATCH_SIZE"], num_workers=8,sampler=sampler,pin_memory=True)
    #Loading the images for test set
    testset = TILS_dataset_Bihead_Area(data_file=test_dataset,labels=[0,1,5,6],labels_tils=[1,2],path=dataset_path,transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=data_hyp["TEST_BATCH_SIZE"],shuffle=True, num_workers=8,pin_memory=True)

    ###########################################
    #  Loss function, optimizers and metrics  #
    ########################################### 
    
    #Loss function
    # LossFun = MultiFocalLoss(alpha=data_hyp['FOCAL_IMP'],gamma=data_hyp['FOCAL_GAMMA'])
    # LossFun = torch.nn.MSELoss()
    LossFun = torch.nn.L1Loss(reduction="sum")
    # LossFun = nn.CrossEntropyLoss().cuda(device)
    LossFun = LossFun.to(device)
    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=data_hyp['LEARNINGRATE'], weight_decay = data_hyp['LAMBDA'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=data_hyp["PATIENCE"])
    #Metrics
    MetricFun = torchmetrics.MetricCollection([torchmetrics.R2Score(dist_sync_on_step=multi_gpu)])
    #                                         torchmetrics.F1Score(num_classes=NUM_CLASSES,average=None,dist_sync_on_step=multi_gpu)])
    # metrics = torchmetrics.MetricCollection([DiceScore(labels=[0,1,2],dist_sync_on_step=multi_gpu)])
    
    TrainMetricDict_cell = MetricFun.clone(prefix='train_cell_').to(device)
    TrainMetricDict_tissue = MetricFun.clone(prefix='train_tissue_').to(device)
    TestMetricDict_cell = MetricFun.clone(prefix='test_cell_').to(device)
    TestMetricDict_tissue = MetricFun.clone(prefix='test_tissue_').to(device)
    TrainMetricDict = [TrainMetricDict_cell,TrainMetricDict_tissue]
    TestMetricDict = [TestMetricDict_cell,TestMetricDict_tissue]
    # DiceScore = torchmetrics.functional.dice_score
    # dicescore = DiceScore(labels=[0,1,2])
    # dicescore = compute_meandice_multilabel
    best_metric = -10000
    ###################################
    #           Training              #
    ###################################    
    if is_wandb:
        wandb.watch(model, log='all') 

    if resume_location is not None:
        model,optimizer,scheduler,epoch_start,best_metric = load_checkpoint(resume_location,model,optimizer,scheduler)
        # best_metric = best_metric.cpu().numpy()
        print("Resuming from saved checkpoint...")
    else:
        epoch_start = 0
    
    for epoch in range(epoch_start,data_hyp['EPOCHS']):
        print("EPOCH: {}".format(epoch))
        train(model, train_loader ,optimizer, LossFun,TrainMetricDict)
        metric,test_loss = test(model,test_loader,LossFun,epoch,TestMetricDict)
        ### Saving model checkpoints
        if is_save and (epoch+1) % 6 == 0:
            save_checkpoint(epoch,model,optimizer,scheduler,metric)
            best_metric = metric
        print(f"Metric at save: {best_metric}")
        scheduler.step(test_loss)
        if is_wandb:
            wandb.log({"Learning rate":optimizer.param_groups[0]["lr"]})
    print("Training done...")
