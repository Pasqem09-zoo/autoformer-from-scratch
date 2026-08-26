"""
Prepare the Electricity dataset.

Original file:
    data/raw/electricity/LD2011_2014.txt

Output file:
    data/raw/electricity/electricity.csv
"""

import pandas as pd


input_path = "data/raw/electricity/LD2011_2014.txt"
output_path = "data/raw/electricity/electricity.csv"


print("Reading original Electricity dataset...")

df = pd.read_csv(
    input_path,
    sep=";",
    decimal=",",
    quotechar='"'
)

# The first column has an empty name in the original file.
# We rename it to "date" to match our dataloader format.
df = df.rename(columns={df.columns[0]: "date"})

df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:", df.shape)
print("First columns:", df.columns[:5].tolist())
print("First date:", df["date"].iloc[0])
print("Last date:", df["date"].iloc[-1])

print("Saving cleaned dataset...")

df.to_csv(output_path, index=False)

print("Saved to:", output_path)