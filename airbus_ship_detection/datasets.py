import os
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread
from .configs import DIR_DATA_TRAIN_IMG as path_train
from .configs import DIR_DATA_TEST_IMG as path_test
from .processing import masks_as_image

class AirbusDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        grp = df.group_by("ImageId").agg([pl.col("EncodedPixels")])
        self.image_ids = grp["ImageId"].to_list()
        self.image_masks = grp["EncodedPixels"].to_list()
        self.transform = transform
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  # use mean and std from ImageNet 

    def __len__(self):
        return len(self.image_ids)
               
    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if (self.mode == 'train') | (self.mode == 'validation'):
            rgb_path = os.path.join(path_train, img_file_name)
        else:
            rgb_path = os.path.join(path_test, img_file_name)
        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[idx])
        
        if self.transform is not None: 
            img, mask = self.transform(img, mask)
            
        if (self.mode == 'train') | (self.mode == 'validation'):
            return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()  
        else:
            return self.img_transform(img), str(img_file_name)