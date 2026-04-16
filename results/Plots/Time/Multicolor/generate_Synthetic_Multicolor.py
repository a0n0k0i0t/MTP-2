import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Excel file
file_path = '../../../Time/Results_difference_Synthetic_Multicolor.xlsx'
df = pd.read_excel(file_path)

# --- Clean all column names ---
# Removes leading/trailing spaces and hidden newline characters
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

print("Detected Columns:", df.columns.tolist())

# 2. Clean the Data
# Forward fill 'Range' (instead of Initial Range) to ensure every row has a value
if 'Range' in df.columns:
    df['Range'] = df['Range'].ffill()
else:
    print("\nWARNING: 'Range' column not found. Check the printed columns above.")

# Clean the Epsilon column to remove hidden characters and convert to numeric
if 'Epsilon' in df.columns:
    df['Epsilon'] = df['Epsilon'].astype(str).str.replace('_x000d_', '', regex=False)
    df['Epsilon'] = df['Epsilon'].str.replace('\r', '', regex=False).str.strip()
    df['Epsilon'] = pd.to_numeric(df['Epsilon'], errors='coerce')

# Drop any completely empty rows that might have resulted from Excel spacing
df = df.dropna(subset=['Epsilon'])

# 3. Filter for your specific Range
target_range = '10000 - 30000'

# Strip any hidden whitespace from the column data before matching
df['Range'] = df['Range'].astype(str).str.strip()
df_filtered = df[df['Range'] == target_range].copy()

# Sort the filtered data sequentially by Epsilon
df_filtered = df_filtered.sort_values(by='Epsilon')

# Ensure Similarity is numeric 
if 'Similarity' in df_filtered.columns:
    df_filtered['Similarity'] = pd.to_numeric(df_filtered['Similarity'], errors='coerce')

# 4. Create the plot
plt.figure(figsize=(10, 6))

# Plot Time (Brute Force) ms
plt.plot(df_filtered['Epsilon'], df_filtered['Time (Brute Force) ms'], 
         marker='o', linestyle='-', color='red', label='Time (Brute Force) ms')

# Plot Time (Range Tree) ms
plt.plot(df_filtered['Epsilon'], df_filtered['Time (Range Tree) ms'], 
         marker='s', linestyle='-', color='blue', label='Time (Range Tree) ms')

# 5. Formatting
plt.title(f'Execution Time Comparison\n(Range: {target_range})')
plt.xlabel('Epsilon & Corresponding Similarity')
plt.ylabel('Time (ms)')

# Use log scale for the x-axis so the epsilon points don't cluster together
plt.xscale('log')

# --- Add Similarity to the X-Axis Labels ---
x_ticks = df_filtered['Epsilon'].values
x_labels = [f"Eps: {eps}\nSim: {sim:.3f}" for eps, sim in zip(df_filtered['Epsilon'], df_filtered['Similarity'])]

# Set the custom ticks and labels on the X-axis
plt.xticks(x_ticks, x_labels)
# -------------------------------------------

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# 6. Show and save
# Note: bbox_inches='tight' ensures the multiline x-labels don't get cut off when saving
save_filename = f"Synthetic_Mulicolor_Time({target_range}).png"
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved successfully as: {save_filename}")