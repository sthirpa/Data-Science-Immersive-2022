import os
import numpy as np
import pandas as pd

# Create the output folder if it does not exist.
if not os.path.exists("output"):
    os.mkdir("output")

# Input data file paths.
files = os.listdir("data")

# PLU code dictionary
plu = pd.read_csv("plu-codes.csv")

# Input-Output dictionary
city_dict = {
    "Atlanta": "atl.csv",
    "Austin": "atx.csv",
    "Boston": "bos.csv",
    "Chicago": "chi.csv",
    "Denver": "den.csv",
    "Los Angeles": "lax.csv",
    "New York": "nyc.csv",
    "San Francisco": "sf.csv",
    "Seattle": "sea.csv",
    "Washington, DC": "dc.csv"
}

# Accepts file and city, returns processed data.
def process_data(file, city):
    df = pd.read_excel("data/" + file, sheet_name=city)
    df["price_usd"] = 1.1 * df["price_eu"]
    df["weight_lb"] = 2.2 * df["weight_kg"]
    df = pd.merge(df, plu, how="left", left_on="prodcode", right_on="plu_code")
    df.drop(["price_eu", "weight_kg", "plu_code"], axis=1, inplace=True)
    df["date"] = file.replace(".xlsx", "")
    return df

# Process data for each city and export to csv
print("Processing data for export...")
for city, file_out in city_dict.items():
    data_list = [process_data(file, city) for file in files]
    data_full = pd.concat(data_list, axis=0)
    data_full["city"] = city
    data_full = data_full.loc[:, 
        ["city", "date", "product", "prodcode", "quantity", "weight_lb", "price_usd"]
    ]
    print(f"Exporting {city} data...")
    data_full.to_csv("output/" + file_out, index=False)
