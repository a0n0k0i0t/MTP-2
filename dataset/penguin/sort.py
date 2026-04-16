import pandas as pd

# 1. Load the dataset (replace with your actual filename)
df = pd.read_csv('extracted_penguins.csv')

# 2. Sort the data based on the 'bill_length_mm' column
# (It sorts in ascending order by default. Add ascending=False inside the parentheses for descending)
df_sorted = df.sort_values(by='bill_length_mm')

# 3. Reorder the columns to make 'bill_length_mm' first and 'species' second
df_reordered = df_sorted[['bill_length_mm', 'species']]

# Print the result to verify
print(df_reordered.head(10))

# 4. Save the final result to a new CSV
df_reordered.to_csv('sorted_penguins.csv', index=False)
print("\nData successfully sorted and saved to 'sorted_penguins.csv'")