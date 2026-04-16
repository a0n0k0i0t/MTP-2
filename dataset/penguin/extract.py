import pandas as pd

# 1. Load the dataset
# Replace 'penguins.csv' with the actual name of your file
df = pd.read_csv('palmerpenguins_extended.csv')

# 2. Extract just the species and bill_length_mm columns
extracted_data = df[['species', 'bill_length_mm']]

# Print the first few rows to verify
print(extracted_data.head())

# 3. Optional: Save the extracted data to a new CSV file
extracted_data.to_csv('extracted_penguins.csv', index=False)
print("\nData successfully saved to 'extracted_penguins.csv'")