# MATH7013 Project
# Data (https://github.com/FutureSharks/financial-data) provided by FutureSharks
# Program for preprocessing a chosen dataset
# Implemented by Joel Thomas

import sys
import datetime as dt
import numpy as np
import pandas as pd
import pyfinancialdata as pyfd

asset = sys.argv[1]
year = int(sys.argv[2])
print(year)
df = pyfd.get(provider="histdata", instrument=asset, year=year)["price"]

# Preprocess the dataset - fill in missing values
i = 0
progress = 0
while i < len(df) - 1:
    curr_date = df.index[i]
    next_date = df.index[i+1]
    
    # Only forward fill missing data for gaps of less than 90 minutes (i.e. ignores weekends and public holidays)
    if (next_date - curr_date).total_seconds()/60 <= 90:
        # Check whether next minute's data already exists
        new_date = curr_date + dt.timedelta(minutes=1)
        if next_date != new_date:
            # Ignore daily trading halt periods (16:15 - 16:30 pm and 17:00 - 18:00 pm)
            if curr_date.time() != dt.time(hour=16, minute=15) and curr_date.time() != dt.time(hour=17, minute=0):
                # Use NaN for now
                new_row = {"price": np.nan}
                new_row = pd.Series(new_row, index=[new_date])
                new_row.index.names = ["date"]
                # Concatenate the new row into the existing dataset
                df = pd.concat([df.iloc[:i+1], new_row, df.iloc[i+1:]])
    
    # Track progress
    if i/(len(df) - 1) >= progress:
        print(f"Progress: {int(progress * 100)}%")
        progress += 0.05
    
    i += 1
    
print(f"Progress: {int(progress * 100)}%")

# Forward fill all NaNs
df.ffill(inplace=True)

# Generate remaining features - raw price changes of the last 45 min and momentum change to the previous 3 hours, 5 hours, 1 day, 3 days, and 10 days
# Convert series object to dataframe and rename column
df = df.to_frame(name="p_t")

df["z_t"] = df["p_t"].diff(periods=1)
for i in range(1, 45):
    col_name = f"z_(t-{i})"
    df[col_name] = df["z_t"].shift(periods=i)
    
for i in [3, 5]:
    lag = i*60
    col_name = f"m_(t-{lag})"
    df[col_name] = df["p_t"].diff(periods=lag)
    
for i in [1, 3, 10]:
    lag = i*24*60
    col_name = f"m_(t-{lag})"
    df[col_name] = df["p_t"].diff(periods=lag)

# Drop all rows that contain NaNs
df.dropna(inplace=True)
df.head()

# Save the final dataset
df.to_csv(f"{asset}_{year}.csv")
