import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data', 'driving_log.csv')

df = pd.read_csv(CSV_PATH, header=None)
df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

print(f'Total samples: {len(df)}')
print(f'Steering range: {df["steering"].min():.4f} to {df["steering"].max():.4f}')

# --- Balancing ---
SAMPLES_PER_BIN = 800
NUM_BINS = 25

counts, bin_edges = np.histogram(df['steering'], bins=NUM_BINS)
remove_indices = []

for i in range(NUM_BINS):
    bin_mask = (df['steering'] >= bin_edges[i]) & (df['steering'] < bin_edges[i + 1])
    bin_indices = df[bin_mask].index.tolist()
    if len(bin_indices) > SAMPLES_PER_BIN:
        drop = np.random.choice(bin_indices, size=len(bin_indices) - SAMPLES_PER_BIN, replace=False)
        remove_indices.extend(drop)

df_balanced = df.drop(index=remove_indices).reset_index(drop=True)
print(f'Samples after balancing: {len(df_balanced)}')

# --- Side-by-side histograms ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df['steering'], bins=NUM_BINS, color='steelblue', edgecolor='black')
axes[0].set_title('Steering Angle Distribution (raw)')
axes[0].set_xlabel('Steering Angle')
axes[0].set_ylabel('Count')

axes[1].hist(df_balanced['steering'], bins=NUM_BINS, color='steelblue', edgecolor='black')
axes[1].set_title(f'Steering Angle Distribution (balanced, cap={SAMPLES_PER_BIN})')
axes[1].set_xlabel('Steering Angle')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
