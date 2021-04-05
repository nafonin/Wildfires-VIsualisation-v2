import pandas as pd

df = pd.read_csv("data.csv")
df = df[["FIRE_YEAR", "DISCOVERY_DOY", "STAT_CAUSE_CODE", "STAT_CAUSE_DESCR", "LONGITUDE", "LATITUDE",
             "FIRE_SIZE", "FIRE_SIZE_CLASS", "STATE", "COUNTY", "Shape"]]
df = df.loc[df["FIRE_YEAR"] >= 2010]
df.to_csv("data compressed since 2010.csv", compression="gzip")