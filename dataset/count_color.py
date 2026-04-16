import pandas as pd

# Load the previously processed dataset
# If you skipped saving it earlier, let me know and I can provide the full combined script!
file_path = 'sorted_age_sex_race.csv'
df = pd.read_csv(file_path)

# Count the occurrences of each category in the 'race' column
race_counts = df['race'].value_counts()

# Extract specific counts (using .get() to avoid errors if a category happens to be 0)
num_white = race_counts.get('White', 0)
num_black = race_counts.get('Black', 0)
num_asian = race_counts.get('Asian', 0)

# Display the results
print("Race Demographics Count:")
print(f"White: {num_white}")
print(f"Black: {num_black}")
print(f"Asian: {num_asian}")

# If you also want to see how many fell into the 'NA' category:
num_na = race_counts.get('NA', 0)
print(f"Other/NA: {num_na}")