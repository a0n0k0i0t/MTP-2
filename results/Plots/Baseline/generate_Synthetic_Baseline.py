import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Excel file
file_path = '../../Baseline/Results_baseline_Synthetic.xlsx'
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

# 2. Clean the Data
if 'Initial Range' in df.columns:
    df['Initial Range'] = df['Initial Range'].ffill()

if 'Epsilon' in df.columns:
    df['Epsilon'] = df['Epsilon'].astype(str).str.replace('_x000d_', '', regex=False)
    df['Epsilon'] = df['Epsilon'].str.replace('\r', '', regex=False).str.strip()
    df['Epsilon'] = pd.to_numeric(df['Epsilon'])

# 3. Filter and Sort
target_range = '10000 - 30000'
df['Initial Range'] = df['Initial Range'].astype(str).str.strip()
df_filtered = df[df['Initial Range'] == target_range].copy()
df_filtered = df_filtered.sort_values(by='Epsilon')

# 4. Create the plot with subplots to allow for dual axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- LEFT Y-AXIS (Ratio Delta on Log Scale) ---
color1 = 'blue'
ax1.set_xlabel('Epsilon (Log Scale)')
ax1.set_ylabel('Ratio delta (Fair)', color=color1)
line1 = ax1.plot(df_filtered['Epsilon'], df_filtered['Ratio delta (Fair)'], 
                 marker='o', linestyle='-', color=color1, label='Ratio delta (Fair)')
ax1.tick_params(axis='y', labelcolor=color1)

# Set log scales for both X and the Left Y axis
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.3)

# --- RIGHT Y-AXIS (Similarity on Linear Scale) ---
ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
color2 = 'green'
ax2.set_ylabel('Similarity (Difference)', color=color2)
line2 = ax2.plot(df_filtered['Epsilon'], df_filtered['Similarity (Difference)'], 
                 marker='s', linestyle='-', color=color2, label='Similarity (Difference)')
ax2.tick_params(axis='y', labelcolor=color2)

# 5. Formatting and Legends
plt.title(f'Impact of Epsilon on Fairness and Similarity\n(Initial Range: {target_range})')

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right')

# 6. Show and save
save_filename = f"Synthetic_Baseline({target_range}).png"
plt.tight_layout() # Ensures labels don't get cut off
plt.savefig(save_filename, dpi=300)
plt.show()

print(f"Plot saved successfully as: {save_filename}")