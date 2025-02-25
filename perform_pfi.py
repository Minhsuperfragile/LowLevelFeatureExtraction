from utils.lowlevelfeatures import *
from model.models import SimpleNeuralNetwork
from utils.constant import param_list
from utils.PFI import permutation_feature_importance, per_class_accuracy
import pandas as pd
import torch

def permutation_score(pred, labels):
    ca = per_class_accuracy(pred, labels)
    return ca[1] + ca[0]

data_folder = 'vdcd_data'
name_format = "test_{name}.csv"

for param in param_list:
    llf = LowLevelFeatureExtractor(**param)
    llf_name = llf.function.__name__

    dataset = pd.read_csv(f'./{data_folder}/{llf_name}/{name_format.format(name = llf_name)}').drop(['image'], axis='columns')
    features_size = dataset.shape[1] - 1

    checkpoint = torch.load(f"./ckpts/model_{llf_name}.pth", map_location='cpu')
    model = SimpleNeuralNetwork(inputs= features_size, classes=pd.unique(dataset.iloc[:, 0]).shape[0])
    model.load_state_dict(checkpoint)
    model.eval()

    features_name = [f'feature_{i}' for i in range(features_size)]

    result = permutation_feature_importance(model, dataset.iloc[:, 1:], dataset.iloc[:, 0], metric=permutation_score)

    result_df = pd.DataFrame(result, index=features_name, columns=[llf_name])
    # print(result_df)
    result_df.sort_values(by=llf_name, ascending=False, inplace=True)

    result_data = result_df.to_csv(index_label=False).splitlines()

    with open('./{data_folder}/pfi_class1_result.txt', 'a') as f:
        f.write('\n')
        for line in result_data:
            f.write(line + '\n')

    print('Result written to ./{data_folder}/pfi_result.txt')