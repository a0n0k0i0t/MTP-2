import random
import csv

num_rows = 30000
filename = "synthetic_30k_unique.csv"
categories = ["1", "2", "3"]

# Generate a list of perfectly unique numbers from 1 to 30,000
unique_indices = list(range(1, num_rows + 1))

# Shuffle the list so the C++ code has to actively sort them when reading
random.shuffle(unique_indices)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=' ') 
    
    for index in unique_indices:
        color_val = random.choice(categories)
        writer.writerow([index, color_val])

print(f"Successfully generated {num_rows} unique records in '{filename}'")