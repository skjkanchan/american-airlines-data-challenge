import pandas as pd

# load raw data
df = pd.read_csv("data/weather/raw/metars.cache.csv")

# keep only what you need
df = df[[
    "station_id",
    "observation_time",
    "visibility_statute_mi",
    "wind_speed_kt",
    "wx_string",
    "flight_category"
]]

# rename columns
df = df.rename(columns={
    "station_id": "airport",
    "observation_time": "datetime",
    "visibility_statute_mi": "visibility",
    "wind_speed_kt": "wind_speed"
})

# convert numeric columns
df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce")
df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

# storm flag
def storm_flag(wx):
    if isinstance(wx, str) and ("TS" in wx or "RA" in wx or "SN" in wx):
        return 1
    return 0

df["storm"] = df["wx_string"].apply(storm_flag)

# IFR flag
df["ifr"] = (
    (df["visibility"] < 3) |
    (df["flight_category"] == "IFR")
).astype(int)

# severity score
df["severity"] = (
    df["storm"] * 2 +
    (df["wind_speed"] > 20).astype(int) +
    df["ifr"]
)

df["bad_weather"] = (df["severity"] >= 2).astype(int)

# drop unnecessary columns
df = df.drop(columns=["wx_string", "flight_category"])

# drop missing values
df = df.dropna()

# save clean dataset
df.to_csv("data/weather/processed/weather_clean.csv", index=False)

print(df.head())
print("Saved weather_clean.csv")