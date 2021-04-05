import pandas as pd

df = pd.read_csv("data.csv")
df = df[["FIRE_YEAR", "DISCOVERY_DOY", "STAT_CAUSE_CODE", "STAT_CAUSE_DESCR", "LONGITUDE", "LATITUDE",
             "FIRE_SIZE", "FIRE_SIZE_CLASS", "STATE", "COUNTY", "Shape"]]
df.to_csv("data compressed.csv", compression="gzip")