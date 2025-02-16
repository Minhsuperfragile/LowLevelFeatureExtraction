import pandas as pd
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./data/linux/vaynen_train_linux.csv", delimiter=',')
test_df = pd.read_csv("./data/linux/vaynen_test_linux.csv", delimiter=',')
root_folder = "../../skin_data/"

for llf_param in param_list[10:11]:
    llf = LowLevelFeatureExtractor(**llf_param)

    llf_name = llf.function.__name__
    # process_dataframe(test_df, llf, root_folder, save_path=f"./data/{llf_name}/vaynen_test_{llf_name}.csv")
    process_dataframe(train_df, llf, root_folder, save_path=f"./data/{llf_name}/vaynen_train_{llf_name}.csv")