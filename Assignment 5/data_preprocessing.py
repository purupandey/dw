import pandas as pd

# import data
raw_data = pd.read_csv("Assignment 5/traffic_signal_data.csv")

# number of rows
print(" Number of rows : ", len(raw_data))

# number of NANs in each column
raw_data.isnull().sum()

# dropping column with most NANs
raw_data = raw_data.drop(["Midblock Route", "Side 2 Route"], axis=1)

# adding "_" for spaces in columns names
fixed_cols = []
for i in raw_data.columns:
    fixed_cols.append(i.replace(" ", "_"))
print(fixed_cols)

# all lowercase columns
fixed_cols = [x.lower() for x in fixed_cols]
raw_data.columns = fixed_cols

# fix date column
raw_data["activation_date"] = raw_data["activation_date"].apply(lambda x: str.replace(x, "-", "/"))

# save the csv
raw_data.to_csv("Assignment 5/data.csv")
