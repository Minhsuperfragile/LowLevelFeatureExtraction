import pandas as pd
from utils.lowlevelfeatures import *
from utils.tools import MultiprocessingExtractor
from utils.dataset import ToNumpy
from utils.constant import param_list
from torchvision import transforms
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./vdcd_data/vdcd_train.csv", delimiter=',')
test_df = pd.read_csv("./vdcd_data/vdcd_test.csv", delimiter=',')
root_folder = "../viemda/"

image_size_normalize = (384,384)

transform = transforms.Compose([
    transforms.Resize(image_size_normalize),  # Resize images to 128x128
    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
    ToNumpy(),  # Convert to tensor
])

for llf_param in param_list:
    llf = LowLevelFeatureExtractor(**llf_param, image_size=image_size_normalize)
    llf_name = llf.function.__name__

    if not os.path.exists(f"./vdcd_data/{llf_name}"):  # Check if directory exists
        os.makedirs(f"./vdcd_data/{llf_name}")  # Create directory if it doesn't exist

    MultiprocessingExtractor.process_dataframe(test_df, llf, transform, root_folder, save_path=f"./vdcd_data/{llf_name}/test_{llf_name}.csv")
    MultiprocessingExtractor.process_dataframe(train_df, llf, transform, root_folder, save_path=f"./vdcd_data/{llf_name}/train_{llf_name}.csv")