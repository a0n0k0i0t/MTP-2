import pandas as pd

# Load the dataset
# Replace 'adult_numerical.csv' with your actual file path
file_path = 'adult_numerical.csv'
df = pd.read_csv(file_path)

# Extract the 'age' and 'sex' columns
extracted_data = df[['age', 'sex']]

# Sort the extracted data by 'age'
# To sort from oldest to youngest instead, add the argument: ascending=False
sorted_data = extracted_data.sort_values(by='age')

# Display the first few rows of the sorted data
print("Extracted and Sorted Data:")
print(sorted_data.head(10))

# Optional: Save the sorted data to a new CSV file
output_file = 'sorted_age_sex.csv'
sorted_data.to_csv(output_file, index=False)
print(f"\nSorted data saved to {output_file}")