import pandas as pd

# Load datasets
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

# Add labels
fake["label"] = 0   # Fake = 0
true["label"] = 1   # True = 1

# Combine and shuffle
combined = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

# Save as new file
combined.to_csv("combined.csv", index=False)

print("✅ combined.csv created successfully!")
