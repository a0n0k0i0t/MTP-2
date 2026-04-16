import pandas as pd

# Load the file. 
# sep='\s+' handles one or more spaces/tabs. 
# Use sep=',' if it is strictly comma-separated.
# header=None is used because your snippet doesn't have column names.
df = pd.read_csv('UrbanDB.csv', sep='\s+', header=None)

# Grab the last column (index -1)
last_column = df.iloc[:, -1]

# Count the occurrences of each value
counts = last_column.value_counts()

# Fetch counts safely using .get() in case a number doesn't exist in the data
print(f"Number of 1's: {counts.get(1, 0)}")
print(f"Number of 2's: {counts.get(2, 0)}")
print(f"Number of 3's: {counts.get(3, 0)}")