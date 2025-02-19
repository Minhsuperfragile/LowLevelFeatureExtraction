from utils import *

for param in param_list:
    llf = LowLevelFeatureExtractor(**param)
    llf_name = llf.function.__name__

    dataset = pd.read_csv(f'./data/{llf_name}/vaynen_test_{llf_name}_new.csv').drop(['image'], axis='columns')
    features_size = dataset.shape[1] - 1

    checkpoint = torch.load(f"./ckpts/model_{llf_name}.pth", map_location='cpu')
    model = SimpleNeuralNetwork(inputs= features_size, classes=2) 
    model.load_state_dict(checkpoint)
    model.eval()

    features_name = [f'feature_{i}' for i in range(features_size-6)]

    result = permutation_feature_importance(model, dataset)

    result_df = pd.DataFrame(result, index=features_name, columns=[llf_name])
    # print(result_df)
    result_df.sort_values(by=llf_name, ascending=False, inplace=True)

    result_data = result_df.to_csv(index_label=False).splitlines()

    with open('./data/pfi_result.txt', 'a') as f:
        f.write('\n')
        for line in result_data:
            f.write(line + '\n')

    print('Result written to ./data/pfi_result.txt')