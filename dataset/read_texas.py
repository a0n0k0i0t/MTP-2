import pandas as pd

# Load the file. 
# sep='\s+' handles spaces between the columns.
df = pd.read_csv('texas_tribune.csv', sep='\s+', header=None)

# Grab the FIRST column (index 0)
first_column = df.iloc[:, 0]

# Count the occurrences of each value
counts = first_column.value_counts()

# Print results safely using .get()
print(f"Number of 0's: {counts.get(0, 0)}")
print(f"Number of 1's: {counts.get(1, 0)}")
print(f"Number of 2's: {counts.get(2, 0)}")