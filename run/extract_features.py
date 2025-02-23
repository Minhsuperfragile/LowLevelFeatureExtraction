import pandas as pd
from utils.lowlevelfeatures import *
from utils.tools import MultiprocessingExtractor
from utils.dataset import ToNumpy
from utils.constant import param_list
from torchvision import transforms

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./data/linux/vaynen_train_linux.csv", delimiter=',')
test_df = pd.read_csv("./data/linux/vaynen_test_linux.csv", delimiter=',')
root_folder = "../../skin_data/"

image_size_normalize = (384,384)

transform = transforms.Compose([
    transforms.Resize(image_size_normalize),  # Resize images to 128x128
    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
    ToNumpy(),  # Convert to tensor
])

for llf_param in param_list:
    llf = LowLevelFeatureExtractor(**llf_param, image_size=image_size_normalize)

    llf_name = llf.function.__name__
    MultiprocessingExtractor.process_dataframe(test_df, llf, transform, root_folder, save_path=f"./data/{llf_name}/vaynen_test_{llf_name}.csv")
    MultiprocessingExtractor.process_dataframe(train_df, llf, transform, root_folder, save_path=f"./data/{llf_name}/vaynen_train_{llf_name}.csv")