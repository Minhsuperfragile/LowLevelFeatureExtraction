import torch
from torch.utils.data import DataLoader
import os
from utils.lowlevelfeatures import LowLevelFeatureExtractor
from utils.constant import param_list
from utils.dataset import CSVMetadataDataset
from model.models import SimpleNeuralNetwork
from utils.tools import train_model, evaluate_model, FilesProcessor
import pandas as pd

root_folder = "../../skin_data/"
csv_folder = "./vdcd_data/"
ckpt_folder = "./ckpts/"

# Check if the directory already exists
if not os.path.exists(ckpt_folder):
    # Create the directory using mkdir()
    os.makedirs(ckpt_folder)
    print(f"Folder '{ckpt_folder}' created.")

batch_size = 32
num_workers = 4
md = True
result_df = FilesProcessor.create_result_df(column_name=["nn"], feature_name=[feat['function'].__name__ for feat in param_list], n_class=3)

for llf_param_set in param_list:
    llf = LowLevelFeatureExtractor(**llf_param_set)

    llf_name = llf.function.__name__

    if not os.path.exists(os.path.join(csv_folder, llf_name)):
        continue

    train_csv = os.path.join(csv_folder, llf_name, f"train_{llf_name}.csv")
    test_csv = os.path.join(csv_folder, llf_name, f"test_{llf_name}.csv")

    train_dataset = CSVMetadataDataset(csv_file=train_csv, root_dir=root_folder, use_md = md)
    test_dataset = CSVMetadataDataset(csv_file=test_csv, root_dir=root_folder, use_md = md)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SimpleNeuralNetwork(inputs= train_dataset.get_input_size(), classes=3)

    train_model(model, train_dataloader, 30, llf_name)
    result , cm = evaluate_model(model, test_dataloader, llf_name)

    result_df.at[llf_name, f"nn"] = result
    result_df.at[f'{llf_name}_0', f"nn"] = cm[0]
    result_df.at[f'{llf_name}_1', f"nn"] = cm[1]
    result_df.at[f'{llf_name}_2', f"nn"] = cm[2]

    torch.save(model.state_dict(), f'./ckpts/model_{llf_name}.pth')

result_df.to_csv("./ckpts/multiple_result.csv")