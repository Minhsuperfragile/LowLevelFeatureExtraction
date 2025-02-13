import pyfeats
from typing import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


from utils import *

llf = LowLevelFeatureExtractor(function=pyfeats.glcm_features, 
                               params={'ignore_zeros': True}, 
                               features_set=['features_mean', 'features_range'])

root_folder = "/mnt/c/Users/trong/Documents/skin_data"

