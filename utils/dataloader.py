import torch.utils.data as data
import torch
from tqdm import tqdm
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
import albumentations
from joblib import Parallel, delayed
from multiprocessing import Manager

class TILS_dataset(data.Dataset):
    ''' Gets the tissue, cell and til density score in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                labels_tils:list=[],
                path:str="/localdisk3/ramanav/TIL_Patches_v2",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(TILS_dataset,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.labels_tils = labels_tils
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        img_path = Path(self.path/Path("images")/Path(metafile.Name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        mask_path = Path(self.path/Path("labels")/Path(metafile.Name))

        #Loading images and labels
        img = np.load(img_path)
        labels = np.load(mask_path)
        tissue_mask = self._apply_onehot(labels[0])
        cell_mask = labels[1]

        if self.transform!=None:
            img_processed,all_mask = self.process_input(img,tissue_mask,cell_mask)
            if len(self.labels_tils)==0:
                return img_processed,all_mask[:-1]
            til_density = self._calc_tils(all_mask)
            return img_processed,(all_mask,til_density)
        else:
            return img,(np.stack((tissue_mask,cell_mask)),metafile.TILdensity)
    
    def _apply_onehot(self,mask):
        """
        Converts tissue mask into one hot encoding
        """
        y = mask
        #Replace roi class with rest
        y = np.where(y==0,7,y)
        y = y-1
        y_onehot = np.zeros((y.size,7))
        y_onehot[np.arange(y.size), y.ravel().astype(np.int32)] = 1
        y_onehot.shape = y.shape + (7,)
        return y_onehot

    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray,cell_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)
        cell_mask = cell_mask.astype(np.int32)

        if len(self.labels)!=0:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        tissue_target.append(cell_mask)
        out = self.transform(image= img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

    def _calc_tils(self,masks):
        """
        Calculates tils density for the given transformed mask
        """
        TILS_area = 0
        tissue_area = 0
        # for k in range(len(self.labels_tils)):
        for k in self.labels_tils:
            tissue_area += torch.sum(masks[k,:,:])
            TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
        return TILS_area/(tissue_area+0.00001)


class Tissue_dataset(data.Dataset):
    ''' Gets the tissue masks in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                mode:str="training",
                path:str="/localdisk3/ramanav/Seg_Patches_v2",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        '''
        super(Tissue_dataset,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        if self.mode=="training":
            img_path = self.path/metafile.Name
            mask_path = self.path/Path("labels_"+Path(metafile.Name).parent.name[-1])/Path(metafile.Name).name
        else:
            img_path = Path(metafile.Name).parent.parent / Path("test_images") / Path(metafile.Name).name
            mask_path = Path(metafile.Name)
        # if Path(metafile.Name).parent.name!="labels":
        #     img_path = Path(metafile.Name).parent.parent / Path("test_images") / Path(metafile.Name).name
        # else:
        #     img_path = Path(metafile.Name).parent.parent / Path("images") / Path(metafile.Name).name
        # #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        # mask_path = Path(metafile.Name)
        # img_path = Path(self.path/Path("images")/Path(metafile.Name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        # mask_path = Path(self.path/Path("labels")/Path(metafile.Name))
        
        #Loading images and labels
        img = np.load(img_path)
        #The labels already doesnt have roi label
        labels = np.load(mask_path)
        
        if self.transform!=None:
            img_processed,tissue_mask = self.process_input(img,labels)
            return img_processed,tissue_mask
        else:
            return img,labels
    
    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)

        if len(self.labels)!=0:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        out = self.transform(image=img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

class Tissue_dataset_density(data.Dataset):
    ''' Gets the tissue density in a patch with the masks'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                mode:str="training",
                path:str="/localdisk3/ramanav/Seg_Patches_v2",
                labels_dens:list=[],
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        '''
        super(Tissue_dataset_density,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.labels_dens = labels_dens
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def _apply_onehot(mask):
        """
        Converts tissue mask into one hot encoding
        """
        y = mask
        #Replace roi class with rest
        # y = np.where(y==0,7,y)
        # y = y-1
        y_onehot = np.zeros((y.size,7))
        y_onehot[np.arange(y.size), y.ravel().astype(np.int32)] = 1
        y_onehot.shape = y.shape + (7,)
        return y_onehot

    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        # if self.mode=="training":
        #     img_path = self.path/metafile.Name

        #     # mask_path = self.path/Path("labels_"+Path(metafile.Name).parent.name[-1])/Path(metafile.Name).name
        # else:
        #     img_path = Path(metafile.Name).parent.parent / Path("test_images") / Path(metafile.Name).name
        #     mask_path = Path(metafile.Name)
        #     img = np.load(img_path)
        #     #The labels already doesnt have roi label
        #     labels = np.load(mask_path)
        #Loading images and labels
        img_path = self.path/metafile.Name
        data = np.load(img_path)
        img = data[:,:,:3]
        labels = data[:,:,-1]
        labels = self._apply_onehot(labels)
        #The labels already doesnt have roi label
        # labels = np.load(mask_path)
        # labels = np.random.rand(256,256,7)
        
        if self.transform!=None:
            img_processed,tissue_mask = self.process_input(img,labels)
            return img_processed,(tissue_mask,self._calc_area(tissue_mask))
        else:
            return img,(labels,self._calc_area(labels))
    
    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)

        if len(self.labels)!=0:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        out = self.transform(image=img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))
    
    def _calc_area(self,masks):
        """
        Gets the tissue density of given tissues
        """
        h,w =  np.shape(masks[0,:,:])
        tissue_area = torch.sum(masks[self.labels_dens,:,:])/float(h*w)
        return tissue_area

class Cell_dataset(data.Dataset):
    """
    Gets Cell dataset ready
    """
    def __init__(self,
                data_file:pd.DataFrame,
                path:str="/localdisk3/ramanav/Cell_Patches",
                normalization_factor=0.6,
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        normalization_factor: Divide the real cellularity score by this number to get closer estimate to pathologist(callibaration)
        '''
        super(Cell_dataset,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.normalization_factor = normalization_factor

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        # file_name = self.master_files[index].name
        file_name = self.df.iloc[index].Name
        img_path = Path(self.path/Path("images")/Path(file_name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        mask_path = Path(self.path/Path("labels")/Path(file_name))

        #Loading images and labels
        img = np.load(img_path)
        labels = np.load(mask_path)
        #TCGA labels has two types of mask, other than TCGA, there is only one mask
        if len(labels.shape)==3:
            cell_mask = labels[1]
        else:
            cell_mask = labels


        if self.transform!=None:
            img_processed,cell_mask = self.process_input(img,cell_mask)
            cellularity_score = self._calc_cellularity(cell_mask)
            return img_processed,(cell_mask,cellularity_score)
        else:
            return img,(cell_mask,self._calc_cellularity(cell_mask))

    def _calc_cellularity(self,mask):
        """
        Calculates total cell area in a given patch
        """
        cell_area = torch.sum(mask)
        h,w =  np.shape(mask)
        return torch.clamp(cell_area/(self.normalization_factor*float(h*w)),max=1.0)


    def process_input(self,img:np.ndarray,cell_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        cell_mask = cell_mask.astype(np.int32)

        out = self.transform(image= img,mask=cell_mask) 

        return out["image"],out["mask"]


class Tissue_dataset_CPU(data.Dataset):
    ''' Gets the tissue masks in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                mode:str="training",
                path:str="/localdisk3/ramanav/Seg_Patches_v2",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        '''
        super(Tissue_dataset_CPU,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.mode = mode

        #Parallel loading of the dataset
        manager = Manager()
        self.data = manager.list()
        
        print("Loading data into CPU")
        _ = Parallel(n_jobs=4)(delayed(self._load_data)(index) for index in tqdm(range(len(self.df))))

    def _load_data(self,index):
        metafile = self.df.iloc[index]
        if self.mode=="training":
            img_path = self.path/metafile.Name
            mask_path = self.path/Path("labels_"+Path(metafile.Name).parent.name[-1])/Path(metafile.Name).name
        else:
            img_path = Path(metafile.Name).parent.parent / Path("test_images") / Path(metafile.Name).name
            mask_path = Path(metafile.Name)
        
        #Loading images and labels
        img = np.load(img_path)
        #The labels already doesnt have roi label
        labels = np.load(mask_path)
        self.data.append([img,labels])
        return
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        img,labels = self.data[index]
        
        if self.transform!=None:
            img_processed,tissue_mask = self.process_input(img,labels)
            return img_processed,tissue_mask
        else:
            return img,labels
    
    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)

        if len(self.labels)!=0:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        out = self.transform(image=img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

class TILS_dataset_Bihead(data.Dataset):
    ''' Gets the cellularity score and normalized invasive tumour area in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                labels_tils:list=[],
                path:str="/localdisk3/ramanav/TIL_Patches_v2",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(TILS_dataset_Bihead,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.labels_tils = labels_tils
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        IMGDIR = ["images","labels"]
        if (Path(metafile.Name).name[0]=="3") or (Path(metafile.Name).name[0]=="4"):
            IMGDIR = ["images_weight","labels_weight"]
        img_path = Path(self.path/Path(IMGDIR[0])/Path(metafile.Name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        mask_path = Path(self.path/Path(IMGDIR[1])/Path(metafile.Name))

        #Loading images and labels
        img = np.load(img_path)
        labels = np.load(mask_path)
        tissue_mask = self._apply_onehot(labels[0])
        cell_mask = labels[1]

        if self.transform!=None:
            img_processed,all_mask = self.process_input(img,tissue_mask,cell_mask)
            if len(self.labels_tils)==0:
                return img_processed,all_mask[:-1]
            til_density,tissue_density = self._calc_area(all_mask)
            return img_processed,(all_mask,til_density,tissue_density)
        else:
            #Will change that later
            return img,(np.stack((tissue_mask,cell_mask)),metafile.TILdensity)
    
    def _apply_onehot(self,mask):
        """
        Converts tissue mask into one hot encoding
        """
        y = mask
        #Replace roi class with rest
        y = np.where(y==0,7,y)
        y = y-1
        y_onehot = np.zeros((y.size,7))
        y_onehot[np.arange(y.size), y.ravel().astype(np.int32)] = 1
        y_onehot.shape = y.shape + (7,)
        return y_onehot

    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray,cell_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)
        cell_mask = cell_mask.astype(np.int32)

        if len(self.labels)>1:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        tissue_target.append(cell_mask)
        out = self.transform(image= img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

    # def _calc_tils(self,masks):
    #     """
    #     Calculates tils density for the given transformed mask
    #     """
    #     TILS_area = 0
    #     tissue_area = 0
    #     # for k in range(len(self.labels_tils)):
    #     for k in self.labels_tils:
    #         tissue_area += torch.sum(masks[k,:,:])
    #         TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
    #     return TILS_area/(tissue_area+0.00001)

    def _calc_area(self,masks):
        """
        Calculates tils density for the given transformed mask
        """
        # cell_area = torch.sum(mask)
        # h,w =  np.shape(mask)
        # return torch.clamp(cell_area/(self.normalization_factor*float(h*w)),max=1.0)
        h,w = np.shape(masks[-1,:,:])
        TILS_area = 0
        tissue_area = 0
        # for k in range(len(self.labels_tils)):
        TILS_area = torch.sum(masks[-1,:,:])/float(h*w)
        tissue_area = torch.sum(masks[self.labels_tils,:,:])/float(h*w)
        return TILS_area,tissue_area
        # for k in self.labels_tils:
        #     tissue_area += torch.sum(masks[k,:,:])
        #     TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
        # return TILS_area/(tissue_area+0.00001)

class TILS_dataset_Bihead_Area(data.Dataset):
    ''' Gets the relevant tissue area and cell area in relevant tissue in a patch'''
    def __init__(self,
                data_file:pd.DataFrame,
                labels:list=[],
                labels_tils:list=[],
                path:str="/localdisk3/ramanav/TIL_Patches_v2",
                transform=None):
        '''
        data_file: (pd.DataFrame) Contains all the file names and TIL density scoring
        path: (str) Path to TIL density dataset
        transform: (albumentations.core.transforms) Transform object
        labels: What all labels to consider for tissue segmentation
        labels_tils: 
        '''
        super(TILS_dataset_Bihead_Area,self).__init__()
        self.df = data_file.copy()
        self.path = Path(path)
        self.transform = transform
        self.labels = labels
        self.labels_tils = labels_tils
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        metafile = self.df.iloc[index]
        IMGDIR = ["images","labels"]
        if (Path(metafile.Name).name[0]=="3") or (Path(metafile.Name).name[0]=="4"):
            IMGDIR = ["images_weight","labels_weight"]
        img_path = Path(self.path/Path(IMGDIR[0])/Path(metafile.Name))
        #Of dimension 2xHxW where dim=0 is the tissue segmentation mask and dim=1 is cell segmentation
        mask_path = Path(self.path/Path(IMGDIR[1])/Path(metafile.Name))

        #Loading images and labels
        img = np.load(img_path)
        labels = np.load(mask_path)
        tissue_mask = self._apply_onehot(labels[0])
        cell_mask = labels[1]

        if self.transform!=None:
            img_processed,all_mask = self.process_input(img,tissue_mask,cell_mask)
            if len(self.labels_tils)==0:
                return img_processed,all_mask[:-1]
            til_density,tissue_density = self._calc_area(all_mask)
            return img_processed,(all_mask,til_density,tissue_density)
        else:
            #Will change that later
            return img,(np.stack((tissue_mask,cell_mask)),metafile.TILdensity)
    
    def _apply_onehot(self,mask):
        """
        Converts tissue mask into one hot encoding
        """
        y = mask
        #Replace roi class with rest
        y = np.where(y==0,7,y)
        y = y-1
        y_onehot = np.zeros((y.size,7))
        y_onehot[np.arange(y.size), y.ravel().astype(np.int32)] = 1
        y_onehot.shape = y.shape + (7,)
        return y_onehot

    def process_input(self,img:np.ndarray,tissue_mask:np.ndarray,cell_mask:np.ndarray):
        """
        Function for processing and performing data augmentation on the input data. Expects one hot form of tissue mask
        Parameters:
            labels(list): Only select certain labels, to form new one hot encoding
        Returns img and tissue+cell masks with the end dimension signifying cell mask
        """
        img = img.astype(np.float32)
        tissue_mask = tissue_mask.astype(np.int32)
        cell_mask = cell_mask.astype(np.int32)

        if len(self.labels)>1:
            tissue_mask = tissue_mask[:,:,self.labels]
            #If the first three classes are not present i.e 0,1,5 then put it into rest class which is at the end
            tissue_mask[:,:,-1] = tissue_mask[:,:,-1] + np.abs(np.sum(tissue_mask,axis=-1)-1)
        
        tissue_target = [tissue_mask[:,:,j] for j in range(tissue_mask.shape[-1])]
        tissue_target.append(cell_mask)
        out = self.transform(image= img,masks=tissue_target) 

        return out["image"],torch.Tensor(np.stack(out["masks"]))

    # def _calc_tils(self,masks):
    #     """
    #     Calculates tils density for the given transformed mask
    #     """
    #     TILS_area = 0
    #     tissue_area = 0
    #     # for k in range(len(self.labels_tils)):
    #     for k in self.labels_tils:
    #         tissue_area += torch.sum(masks[k,:,:])
    #         TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
    #     return TILS_area/(tissue_area+0.00001)

    def _calc_area(self,masks):
        """
        Calculates tils area in relevant tissue regions with tissue area as well for the given transformed mask
        """
        # cell_area = torch.sum(mask)
        # h,w =  np.shape(mask)
        # return torch.clamp(cell_area/(self.normalization_factor*float(h*w)),max=1.0)
        h,w = np.shape(masks[-1,:,:])
        TILS_area = 0
        tissue_area = 0
        # for k in range(len(self.labels_tils)):
        # TILS_area = torch.sum(masks[-1,:,:])/float(h*w)
        # tissue_area = torch.sum(masks[self.labels_tils,:,:])/float(h*w)
        # return TILS_area,tissue_area
        for k in self.labels_tils:
            tissue_area += torch.sum(masks[k,:,:])
            TILS_area += torch.sum(torch.logical_and(masks[-1,:,:],masks[k,:,:]))
        return TILS_area/float(h*w), tissue_area/float(h*w)