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
md = "md"
# result_df = pd.read_csv("./ckpts/multiple_result.csv", index_col="features_name")

for llf_param_set in param_list:
    llf = LowLevelFeatureExtractor(**llf_param_set)

    llf_name = llf.function.__name__

    if not os.path.exists(os.path.join(csv_folder, llf_name)):
        continue

    train_csv = os.path.join(csv_folder, llf_name, f"vaynen_train_{llf_name}_new.csv")
    test_csv = os.path.join(csv_folder, llf_name, f"vaynen_test_{llf_name}_new.csv")

    train_dataset = CSVMetadataDataset(csv_file=train_csv, root_dir=root_folder, md=(md=="md"))
    test_dataset = CSVMetadataDataset(csv_file=test_csv, root_dir=root_folder, md=(md=="md"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SimpleNeuralNetwork(inputs= train_dataset.get_input_size(), classes=2)

    train_model(model, train_dataloader, llf, 30, llf_name)
    result , cm = evaluate_model(model, test_dataloader, llf, llf_name)

    # result_df.at[llf_name, f"nn_{md}"] = result
    # result_df.at[f'{llf_name}_0', f"nn_{md}"] = cm[0]
    # result_df.at[f'{llf_name}_1', f"nn_{md}"] = cm[1]

    torch.save(model.state_dict(), f'./ckpts/model_{llf_name}.pth')

# result_df.to_csv("./ckpts/multiple_result.csv")