import pandas as pd
import torch 
from sklearn.inspection import permutation_importance
import os
from utils import *
from PFI import *
#define path of something
path='/mnt/e/VAST/Low-level-feature/data/'
root_path="/mnt/e/VAST/Skin_detect/skin_data"
folder_names=[name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
batch_size = 32
num_workers = 4
list_csv=[f'/mnt/e/VAST/Low-level-feature/data/{i}/vaynen_test_{i}.csv' for i in folder_names]
list_model=[f'/mnt/e/VAST/Low-level-feature/ckpts/model_{i}.pth' for i in folder_names]
def model_predict(X):
    with torch.no_grad():
        return model(X).cpu().numpy()
#load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#output txt for file.
output_file='/mnt/e/VAST/Low-level-feature/permutation_imp/feature_important.txt'
#start testing
with open(output_file,'w') as f_out:
    for csv_file, model_file in zip(list_csv,list_model):
        print(f'processing {csv_file}')
        print(f'model: {model_file}')
        checkpoint = torch.load(model_file, map_location=device)
        test_dataset = CSVMetadataDataset(csv_file=csv_file, root_dir=root_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
        sample_batch = next(iter(test_loader))
        _, metadata_sample, _ = sample_batch  # Extract metadata tensor
        feature_size = metadata_sample.shape[1]
        model = SimpleNeuralNetwork(inputs=feature_size - 6 ).to(device)
        model.load_state_dict(checkpoint)  # Load only the model weights
        model.eval()
        for batch in test_loader:
            _, metadata, label = batch
            feature_names = [f'Feature_{i}' for i in range(metadata.shape[1])]
            metadata_df = pd.DataFrame(metadata.cpu().numpy(), columns=feature_names)
            #define model

            result = permutation_feature_importance(model,device, metadata, label, n_repeats=20)
            f_out.write(f'feature importance for {model_file} \n')
            metadata_df = pd.DataFrame(metadata.numpy(), columns=[f'feature_{i}' for i in range(metadata.shape[1])])
            importance_df=pd.DataFrame({'Feature':metadata_df.columns,'Importance':result})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            for _, row in importance_df.iterrows():
                f_out.write(f"{row['Feature']}: {row['Importance']:.6f}\n")
            f_out.write("-" * 50 + "\n")

print('all done :>')
