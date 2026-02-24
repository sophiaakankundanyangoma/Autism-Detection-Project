import pandas as pd

# Path to your dataset
DATA_PATH = "data/asd.csv"  # make sure this matches your file name

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Quick look at the first 5 rows
print(df.head())
