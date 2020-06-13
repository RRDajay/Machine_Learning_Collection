# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:45:20 2020

@author: ryan_
"""

import os
import torch
import pandas as pd
from skimage import io

class CustomDatasetLoader(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, root_directory, transform=None):
        
        self.annotations = pd.read_csv(csv_file)
        self.root_directory = root_directory
        self.transform = transform 
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        img_path = os.path.join(self.root_directory, self.annotations.iloc[index, 0])
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        image = io.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
        


