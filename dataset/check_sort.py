import csv

def is_csv_sorted(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            # Read the first row (potentially a header)
            # You might want to skip this if you know there is a header
            first_row = next(reader) 
            # If you are sure there is a header, you should use `next(reader)` again 
            # to get the first actual data row
            current_value = first_row[0]
        except StopIteration:
            return True # Empty file is considered sorted

        for row in reader:
            if not row:
                continue # Skip empty lines
            next_value = row[0]
            # Comparison for ascending order. Adjust for numerical comparison if needed.
            if next_value < current_value:
                return False 
            current_value = next_value
    return True

# Example Usage:
filename = 'texas_tribune.csv'
if is_csv_sorted(filename):
    print(f"The file '{filename}' is sorted on the first column.")
else:
    print(f"The file '{filename}' is NOT sorted on the first column.")

