import pandas as pd

# Load the dataset
# Replace 'adult_numerical.csv' with your actual file path
file_path = 'adult_numerical.csv'
df = pd.read_csv(file_path)

# Remove leading/trailing spaces from the 'race' column just in case
if df['race'].dtype == 'object':
    df['race'] = df['race'].str.strip()

# Define the function to filter and rename the race categories
def categorize_race(race):
    if race == 'Black':
        return 'Black'
    elif race == 'White':
        return 'White'
    else:
        return 'Asian'

# Apply the categorization rule to the 'race' column
df['race'] = df['race'].apply(categorize_race)

# Extract the 'age', 'sex', and 'race' columns
extracted_data = df[['age', 'sex', 'race']]

# Sort the extracted data by 'age'
sorted_data = extracted_data.sort_values(by='age')

# Display the first few rows of the sorted data
print("Extracted, Categorized, and Sorted Data:")
print(sorted_data.head(15))

# Optional: Save the sorted data to a new CSV file
output_file = 'sorted_age_sex_race.csv'
sorted_data.to_csv(output_file, index=False)
print(f"\nFinal data saved to {output_file}")