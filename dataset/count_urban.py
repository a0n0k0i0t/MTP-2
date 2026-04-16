from collections import Counter

# Updated to use your specific file name
filename = 'UrbanDB.csv'
counts = Counter()

try:
    with open(filename, 'r') as file:
        for line in file:
            # Skip any empty lines
            if line.strip():
                # NOTE: .split() handles spaces and tabs. 
                # If your actual file uses commas, change this to: .split(',')
                row_data = line.strip().split()
                
                # Grab the last item in the row
                last_item = row_data[-1]
                counts[last_item] += 1

    print(f"--- Counts for {filename} ---")
    # Sort the results so 1s, 2s, 3s appear in order
    for number, count in sorted(counts.items()):
        print(f"{number}s: {count}")

except FileNotFoundError:
    print(f"Error: Could not find '{filename}'.")
    print("Please make sure the file is in the exact same folder as this Python script.")