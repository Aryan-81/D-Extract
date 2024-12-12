import pandas as pd

# Load the original CSV file
df = pd.read_csv('dataset/test.csv')

# Define the number of rows per split file
rows_per_file = 250

# Calculate the number of parts needed
num_files = len(df) // rows_per_file + (1 if len(df) % rows_per_file != 0 else 0)

# Split and save each part
for i in range(num_files):
    start_row = i * rows_per_file
    end_row = start_row + rows_per_file
    part_df = df[start_row:end_row]
    
    # Save the split file
    part_df.to_csv(f'parts/part_{i+1}.csv', index=False)

print(f"CSV file split into {num_files} parts, each with up to {rows_per_file} rows.")
