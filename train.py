from torchvision import transforms
from torch.utils.data import DataLoader
import os
from utils import *

root_folder = "../../skin_data/"
csv_folder = "./data/"
ckpt_folder = "./ckpts/"
# Check if the directory already exists
if not os.path.exists(ckpt_folder):
    # Create the directory using mkdir()
    os.makedirs(ckpt_folder)
    print(f"Folder '{ckpt_folder}' created.")

batch_size = 32
num_workers = 4

for llf_param_set in param_list:
    llf = LowLevelFeatureExtractor(**llf_param_set)

    llf_name = llf.function.__name__

    if not os.path.exists(os.path.join(csv_folder, llf_name)):
        continue

    train_csv = os.path.join(csv_folder, llf_name, f"vaynen_train_{llf_name}.csv")
    test_csv = os.path.join(csv_folder, llf_name, f"vaynen_test_{llf_name}.csv")

    train_dataset = CSVMetadataDataset(csv_file=train_csv, root_dir=root_folder)
    test_dataset = CSVMetadataDataset(csv_file=test_csv, root_dir=root_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SimpleNeuralNetwork(inputs= llf.get_features_size())

    train_model(model, train_dataloader, llf, 30, llf_name)
    result = evaluate_model(model, test_dataloader, llf, llf_name)

    with open("./ckpts/results.txt", "a") as f:
        f.write(f"{result}\n")

    torch.save(model.state_dict(), f'./ckpts/model_{llf_name}.pth')