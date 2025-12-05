import os
import pandas as pd

IMG_DIR = "../scratch"  # where your 50 images live

# Read original labels
train = pd.read_csv("train_labels.csv")
val = pd.read_csv("val_labels.csv")

# List of files you actually have locally
local_files = set(os.listdir(IMG_DIR))

# Filter rows where the filename (col 0) is in local_files
train_small = train[train.iloc[:, 0].isin(local_files)]
val_small   = val[val.iloc[:, 0].isin(local_files)]

print("Original train size:", len(train))
print("Original val size:", len(val))
print("Filtered train size:", len(train_small))
print("Filtered val size:", len(val_small))

# Save new CSVs
train_small.to_csv("train_labels_small.csv", index=False)
val_small.to_csv("val_labels_small.csv", index=False)
print("Wrote train_labels_small.csv and val_labels_small.csv")
