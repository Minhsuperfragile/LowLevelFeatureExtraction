import pandas as pd
import numpy as np
from PIL import Image
import os
import multiprocessing
from tqdm import tqdm
from utils import *
import time
import warnings
from tabulate import tabulate
warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./data/vaynen_train_linux.csv", delimiter=',')
test_df = pd.read_csv("./data/vaynen_test_linux.csv", delimiter=',')
root_folder = "../../skin_data/"

for llf_param in param_list[3:10]:
    llf = LowLevelFeatureExtractor(**llf_param)

    llf_name = llf.function.__name__
    process_dataframe(test_df, llf, root_folder, save_path=f"./data/{llf_name}/vaynen_test_{llf_name}.csv")
    process_dataframe(train_df, llf, root_folder, save_path=f"./data/{llf_name}/vaynen_train_{llf_name}.csv")