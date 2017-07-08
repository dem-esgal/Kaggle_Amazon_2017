from __future__ import print_function
import torch.utils.data as data
import torch.nn as nn
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import pandas as pd
import fnmatch
from collections import Counter
import numpy as np
from skimage import io
from glob import glob



class LoadTrainFolder(nn.Module):
    
    def __init__(self, dir_train, transform = None):
        super(LoadTrainFolder, self).__init__()
        files = sorted(glob(os.path.join(dir_train, '*.tif')))

        datafile = pd.read_csv('train_v2.csv')

        weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
        land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road']
        rare_labels = ['slash_burn', 'conventional_mine', 'bare_ground',
                       'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
        other_labels = land_labels+rare_labels
        label_unique = weather_labels+land_labels+rare_labels

        index_vector = datafile.tags.str.get_dummies(sep=" ")
        index_vector = index_vector[label_unique].as_matrix().astype(np.float32)


        self.files = files
        self.transform = transform
        self.index_vector = index_vector
        self.nclasses = len(label_unique)
    
    def __getitem__(self, index):

        img = io.imread(self.files[index]).astype(np.float32)/(2**16+0.0-1)
        label = self.index_vector[index]

        if self.transform is not None:
            img = self.transform(img)
        
        
        return img, label
    
    def __len__(self):
        return len(self.files)

class LoadTestFolder(nn.Module):
    
    def __init__(self, dir_test, transform = None):
        super(LoadTestFolder, self).__init__()
        files = sorted(glob(os.path.join(dir_test, '*.tif')))
        
        weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
        land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road']
        rare_labels = ['slash_burn', 'conventional_mine', 'bare_ground',
                       'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
        other_labels = land_labels+rare_labels
        label_unique = weather_labels+land_labels+rare_labels
        

        self.files = files
        self.transform = transform
        self.nclasses = len(label_unique)
    
    def __getitem__(self, index):

        img = io.imread(self.files[index]).astype(np.float32)/(2**16+0.0-1)

        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.files)
