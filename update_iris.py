import pandas as pd
import numpy as np

# Load original iris.csv
df = pd.read_csv("iris.csv")

# Modify: Add small noise to simulate data drift
df['sepal_length'] += np.random.normal(0, 0.15, len(df))

# Optional: shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Overwrite iris.csv directly
df.to_csv("iris.csv", index=False)

print("✅ iris.csv updated successfully.")
