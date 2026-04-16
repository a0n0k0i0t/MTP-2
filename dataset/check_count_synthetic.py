import pandas as pd

filename = 'synthetic_30k_unique.csv'

# Load the data, telling pandas it's separated by spaces and has no header row
df = pd.read_csv(filename, sep=' ', header=None, names=['index', 'color'])

# Count how many times each color appears
color_counts = df['color'].value_counts()

print("--- Category Counts ---")
# Sort the index so it prints 1, 2, 3 in order
print(color_counts.sort_index())