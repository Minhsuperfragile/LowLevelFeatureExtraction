import pandas as pd
import numpy as np
from PIL import Image
import os
import multiprocessing
import time

# Sample DataFrame with image paths and labels
data = {
    "image_path": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"],
    "label": [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Define a custom transformation function using PIL
def extract_features(args):
    """
    Custom function to extract features from an image using PIL.
    Runs in multiple processes.
    """
    image_path, resize_shape = args

    if not os.path.exists(image_path):
        return image_path, np.zeros(resize_shape[0] * resize_shape[1])  # Return zero vector if the image is missing
    
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(resize_shape)  # Resize based on the given shape
    feature_vector = np.array(img).flatten()  # Convert to a 1D array (feature vector)
    
    return image_path, feature_vector

# Multiprocessing worker function
def process_chunk(df_chunk, resize_shape):
    return [extract_features((row["image_path"], resize_shape)) for _, row in df_chunk.iterrows()]

# Number of processes to use
num_processes = multiprocessing.cpu_count() // 2  # Use half of available CPU cores
resize_shape = (3, 2)  # Resize shape for images

# Split DataFrame into chunks for multiprocessing
df_chunks = np.array_split(df, num_processes)

# Start time tracking
start_time = time.time()

# Start multiprocessing pool
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.starmap(process_chunk, [(chunk, resize_shape) for chunk in df_chunks])

# Combine all processed results into a DataFrame
flat_results = [item for sublist in results for item in sublist]  # Flatten list of lists
df_features = pd.DataFrame(flat_results, columns=["image_path", "features"])

# Merge features with original DataFrame
df_final = df.merge(df_features, on="image_path")

# Expand feature vector into separate columns
feature_columns = [f'feature_{i}' for i in range(resize_shape[0] * resize_shape[1])]
df_feature_values = pd.DataFrame(df_final["features"].to_list(), columns=feature_columns)

# Merge feature columns into the original dataframe
df_final = df_final.drop(columns=["features"]).reset_index(drop=True)
df_final = pd.concat([df_final, df_feature_values], axis=1)

# End time tracking
end_time = time.time()
execution_time = end_time - start_time

# Display the final DataFrame
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Multiprocessing Image Data Processing", dataframe=df_final)

# Print completion message
print(f"✅ Feature extraction completed in {execution_time:.2f} seconds.")
