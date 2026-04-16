import pandas as pd

# 1. Load the dataset 
# (Change to 'extracted_penguins.csv' if you want to use the original extracted file)
filename = 'sorted_penguins.csv'
df = pd.read_csv(filename)

# 2. Count the occurrences of each unique value in the 'species' column
species_counts = df['species'].value_counts()

# 3. Print the result
print("--- Species Count ---")
print(species_counts)