import pandas as pd

df = pd.read_csv("data/weather/raw/metars.cache.csv")

print(df.head())
print(df.columns)