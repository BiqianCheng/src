import pandas as pd
import os

# Get the directory of the CSV file
dir_path = os.path.dirname(os.path.realpath('testset_preds.csv'))

# Read the CSV file
df = pd.read_csv('testset_preds.csv')

# Filter the data
filtered_df = df[(df['label'] == 1) & (df['prediction'] == 0)]

# Define the output file path
output_file_path = os.path.join(dir_path, 'filtered_data.txt')

# Store the filtered data into a text file
filtered_df.to_csv(output_file_path, index=False)