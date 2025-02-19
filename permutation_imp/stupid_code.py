import pandas as pd
import torch 
from sklearn.inspection import permutation_importance
import os
from utils import *
from PFI import *
from tqdm import tqdm
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
output_file='/mnt/e/VAST/Low-level-feature/permutation_imp/feature_important_2.txt'
#start testing
with open(output_file, 'w') as f_out:
    for csv_file, model_file in tqdm(list(zip(list_csv, list_model)), total=len(list_csv), desc="Models"):
        print(f'Processing {csv_file}')
        print(f'Model: {model_file}')
        
        # Load model weights
        checkpoint = torch.load(model_file, map_location=device)
        test_dataset = CSVMetadataDataset(csv_file=csv_file, root_dir=root_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Get feature dimension info from one sample
        sample_batch = next(iter(test_loader))
        _, metadata_sample, _ = sample_batch  
        feature_size = metadata_sample.shape[1]

        model = SimpleNeuralNetwork(inputs=feature_size - 6).to(device)
        model.load_state_dict(checkpoint)
        model.eval()

        # Accumulate the entire dataset so we compute permutation importance once per model.
        all_metadata = []
        all_labels = []
        for batch in test_loader:
            _, metadata, label = batch
            all_metadata.append(metadata)
            all_labels.append(label)
        all_metadata = torch.cat(all_metadata, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create feature names
        feature_names = [f'Feature_{i}' for i in range(all_metadata.shape[1])]
        
        # Compute permutation importance over the entire dataset
        result = permutation_feature_importance(model, device, all_metadata, all_labels, n_repeats=20)
        if isinstance(result, torch.Tensor):
                result = result.cpu().numpy().flatten()

            # Validate shape
        if len(result) != all_metadata.shape[1]:
                raise ValueError("Mismatch: Feature importance result has incorrect dimensions.")
            
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        f_out.write(f'Feature importance for {model_file}\n')
        for _, row in importance_df.iterrows():
                f_out.write(f"{row['Feature']}: {row['Importance']:.6f}\n")
        f_out.write("-" * 50 + "\n")

print('all done :>')
