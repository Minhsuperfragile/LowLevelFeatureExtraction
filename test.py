from utils.tools import FilesProcessor, get_current_datetime
from utils.constant import param_list

columns = ['nn_md', 'nn_nmd', 'random_forest_md', 'random_forest_nmd', 
           'decision_tree_md', 'decision_tree_nmd', 'gradient_boost_md', 'gradient_boost_nmd','xgboost_md', 'xgboost_nmd']
features_name = [f'{i["function"].__name__}' for i in param_list]

# print(FilesProcessor.create_result_df(columns, features_name, n_class=2))
# print(get_current_datetime())

# FilesProcessor.generate_csv_from_folder('../viemda/train', save_name='./vdcd_data/vdcd_train.csv')
# FilesProcessor.generate_csv_from_folder('../viemda/test', save_name='./vdcd_data/vdcd_test.csv')

import pandas as pd

df = pd.read_csv('./vdcd_data/vdcd_test.csv')
df['image'] = df['image'].apply(lambda x: x[10:])
df.to_csv('./vdcd_data/vdcd_test.csv', index=False)