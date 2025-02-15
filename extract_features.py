from utils import *
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # Surpress Numpy Warning 

train_df = pd.read_csv("./data/vaynen_train_linux.csv", delimiter=',')
test_df = pd.read_csv("./data/vaynen_test_linux.csv", delimiter=',')
root_folder = "../../skin_data/"

# Function to process an image path
def extract_features(image_path: os.PathLike, root_folder: os.PathLike, llf: LowLevelFeatureExtractor) -> np.ndarray:
    image_path = os.path.join(root_folder, image_path)
    image = Image.open(image_path).convert('L') # Convert to gray-scale
    image = np.array(image)
    features = llf.process_single_image(image)
    return features

# Function to process a smaller dataframe chunk
def process_dataframe_chunk(df_chunk: pd.DataFrame, root_folder: os.PathLike, llf: LowLevelFeatureExtractor):
    df_chunk["features"] = df_chunk["image"].apply(lambda image_path: extract_features(image_path, root_folder, llf))
    return df_chunk

# Number of part to be split in 
X = 4
train_df_chunks = np.array_split(train_df, X)
test_df_chunks = np.array_split(test_df, X)

for llf_param in param_list[:1]:
    llf = LowLevelFeatureExtractor(**llf_param)
    n_features = llf.get_features_size()

    start_time = time.time()

    #region Test data process
    # Use ThreadPoolExecutor for parallel processing with multiple arguments
    results = []
    with ThreadPoolExecutor(max_workers=X) as executor:
        futures = [executor.submit(process_dataframe_chunk, chunk, root_folder, llf) for chunk in test_df_chunks]
        for future in futures:
            results.append(future.result())

    # Combine all processed chunks back into a single DataFrame
    df_final = pd.concat(results, ignore_index=True)

    # Expand feature vector into separate columns
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    df_features = pd.DataFrame(df_final["features"].to_list(), columns=feature_columns)

    # Merge feature columns into the original dataframe
    df_final = df_final.drop(columns=["features"]).reset_index(drop=True)
    df_final = pd.concat([df_final, df_features], axis=1)

    df_final.to_csv(f"./data/vaynen_test_{llf.function.__name__}.csv", index=False)
    #endregion

    #region Train data process
    # Use ThreadPoolExecutor for parallel processing with multiple arguments
    results = []
    with ThreadPoolExecutor(max_workers=X) as executor:
        futures = [executor.submit(process_dataframe_chunk, chunk, root_folder, llf) for chunk in train_df_chunks]
        for future in futures:
            results.append(future.result())

    # Combine all processed chunks back into a single DataFrame
    df_final = pd.concat(results, ignore_index=True)

    # Expand feature vector into separate columns
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    df_features = pd.DataFrame(df_final["features"].to_list(), columns=feature_columns)

    # Merge feature columns into the original dataframe
    df_final = df_final.drop(columns=["features"]).reset_index(drop=True)
    df_final = pd.concat([df_final, df_features], axis=1)

    df_final.to_csv(f"./data/vaynen_train_{llf.function.__name__}.csv", index=False)
    #endregion

    end_time = time.time()  # End timer\
    seconds = end_time - start_time

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    print(f"Execution time: {hours} hours, {minutes} minutes, {remaining_seconds} seconds for {llf.function.__name__}")