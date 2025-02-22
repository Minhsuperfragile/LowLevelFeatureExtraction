from utils.tools import FilesProcessor
from utils.constant import param_list

columns = ['nn_md', 'nn_nmd', 'random_forest_md', 'random_forest_nmd', 
           'decision_tree_md', 'decision_tree_nmd', 'gradient_boost_md', 'gradient_boost_nmd','xgboost_md', 'xgboost_nmd']
features_name = [f'{i["function"].__name__}' for i in param_list]

print(FilesProcessor.create_result_df(columns, features_name, n_class=2))