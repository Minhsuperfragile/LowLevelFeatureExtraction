import pandas as pd
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./data/linux/vaynen_train_linux.csv", delimiter=',')
test_df = pd.read_csv("./data/linux/vaynen_test_linux.csv", delimiter=',')
root_folder = "../../skin_data/"

transform = transforms.Compose([
    transforms.Resize((386,386)),  # Resize images to 128x128
    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
    ToNumpy(),  # Convert to tensor
])

for llf_param in param_list[22:]:
    llf = LowLevelFeatureExtractor(**llf_param)

    llf_name = llf.function.__name__
    process_dataframe(test_df, llf, transform, root_folder, save_path=f"./data/{llf_name}/vaynen_test_{llf_name}.csv")
    process_dataframe(train_df, llf, transform, root_folder, save_path=f"./data/{llf_name}/vaynen_train_{llf_name}.csv")