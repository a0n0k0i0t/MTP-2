import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Excel file
file_path = '../Baseline/Results_baseline_UrbanDB.xlsx'
df = pd.read_excel(file_path)

# --- Clean all column names ---
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

# 2. Clean the Data
if 'Initial Range' in df.columns:
    df['Initial Range'] = df['Initial Range'].ffill()

if 'Epsilon' in df.columns:
    df['Epsilon'] = df['Epsilon'].astype(str).str.replace('_x000d_', '', regex=False)
    df['Epsilon'] = df['Epsilon'].str.replace('\r', '', regex=False).str.strip()
    df['Epsilon'] = pd.to_numeric(df['Epsilon'])

# 3. Filter for your specific Initial Range
target_range = '- 0.081253 to -0.053671'

df['Initial Range'] = df['Initial Range'].astype(str).str.strip()
df_filtered = df[df['Initial Range'] == target_range].copy()

# Sort the filtered data sequentially by Epsilon
df_filtered = df_filtered.sort_values(by='Epsilon')

# 4. Create the plot
plt.figure(figsize=(10, 6))

# Plot Ratio delta (Fair)
plt.plot(df_filtered['Epsilon'], df_filtered['Ratio delta (Fair)'], 
         marker='o', linestyle='-', color='blue', label='Ratio delta (Fair)')

# Add actual values above the points (Ratio delta)
for x, y in zip(df_filtered['Epsilon'], df_filtered['Ratio delta (Fair)']):
    # y + 0.02 adds a small offset so the text isn't overlapping the dot
    plt.text(x, y + 0.02, f'{y:.3f}', color='blue', ha='center', va='bottom', fontsize=9)

# Plot Similarity (Difference)
plt.plot(df_filtered['Epsilon'], df_filtered['Similarity (Difference)'], 
         marker='s', linestyle='-', color='green', label='Similarity (Difference)')

# Add actual values below the points (Similarity)
for x, y in zip(df_filtered['Epsilon'], df_filtered['Similarity (Difference)']):
    # y - 0.02 adds a small offset downwards
    plt.text(x, y - 0.02, f'{y:.3f}', color='green', ha='center', va='top', fontsize=9)

# 5. Formatting
plt.title(f'Impact of Epsilon on Fairness and Similarity\n(Initial Range: {target_range})')
plt.xlabel('Epsilon')
plt.ylabel('Metric Value')

# Extend the y-axis slightly so the highest and lowest text labels don't get cut off
plt.ylim(df_filtered[['Ratio delta (Fair)', 'Similarity (Difference)']].min().min() - 0.1, 
         df_filtered[['Ratio delta (Fair)', 'Similarity (Difference)']].max().max() + 0.1)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# 6. Show and save with the dynamic naming format
plt.tight_layout()

# Save the file as Urban(initial_range).png
save_filename = f"Urban({target_range}).png"
plt.savefig(save_filename, dpi=300)
plt.show()

print(f"Plot saved successfully as: {save_filename}")