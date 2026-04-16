import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Excel file
file_path = '../Baseline/Results_baseline_UrbanDB.xlsx'
df = pd.read_excel(file_path)

# --- NEW FIX: Clean all column names ---
# This removes leading/trailing spaces and hidden newline characters from the headers
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

# Print the columns so you can verify they are correct
print("Detected Columns:", df.columns.tolist())
# ---------------------------------------

# 2. Clean the Data
# Forward fill 'Initial Range' to ensure every row has a value assigned
if 'Initial Range' in df.columns:
    df['Initial Range'] = df['Initial Range'].ffill()
else:
    print("\nWARNING: 'Initial Range' still not found. Check the printed columns above to see what it is actually named.")

# Clean the Epsilon column to remove hidden characters and convert to numeric
if 'Epsilon' in df.columns:
    df['Epsilon'] = df['Epsilon'].astype(str).str.replace('_x000d_', '', regex=False)
    df['Epsilon'] = df['Epsilon'].str.replace('\r', '', regex=False).str.strip()
    df['Epsilon'] = pd.to_numeric(df['Epsilon'])

# 3. Filter for your specific Initial Range
target_range = '-0.130983 to 0.228074'

# Strip any hidden whitespace from the column data just to be safe before matching
df['Initial Range'] = df['Initial Range'].astype(str).str.strip()
df_filtered = df[df['Initial Range'] == target_range].copy()

# Sort the filtered data sequentially by Epsilon
df_filtered = df_filtered.sort_values(by='Epsilon')

# 4. Create the plot
plt.figure(figsize=(10, 6))

# Plot Ratio delta (Fair)
plt.plot(df_filtered['Epsilon'], df_filtered['Ratio delta (Fair)'], 
         marker='o', linestyle='-', color='blue', label='Ratio delta (Fair)')

# Plot Similarity (Difference)
plt.plot(df_filtered['Epsilon'], df_filtered['Similarity (Difference)'], 
         marker='s', linestyle='-', color='green', label='Similarity (Difference)')

# 5. Formatting
plt.title(f'Impact of Epsilon on Fairness and Similarity\n(Initial Range: {target_range})')
plt.xlabel('Epsilon')
plt.ylabel('Metric Value')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# 6. Show and save
save_filename = f"Urban({target_range}).png"
plt.savefig(save_filename, dpi=300)
plt.show()

print(f"Plot saved successfully as: {save_filename}")