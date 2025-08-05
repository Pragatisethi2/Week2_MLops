import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: python poison_iris_labels.py input.csv output.csv percent")
    sys.exit(1)

infile, outfile, percent = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(infile)
labels = df['species'].unique()
n_poison = int(len(df) * percent / 100)
idx_to_poison = np.random.choice(df.index, n_poison, replace=False)
for idx in idx_to_poison:
    original = df.loc[idx, "species"]
    choices = [label for label in labels if label != original]
    df.loc[idx, "species"] = np.random.choice(choices)
df.to_csv(outfile, index=False)
print(f"Poisoned {n_poison} rows. Saved as {outfile}")
