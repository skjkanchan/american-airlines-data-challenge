import pandas as pd
import numpy as np

# ── FILE PATHS ───────────────────────────────────────────────────────────────
FLIGHT_FILE  = r"C:\Spring 2026\American Airlines Challenge\Data\BTS Monthly\2025 Jan\T_ONTIME_REPORTING_20260414_165518\T_ONTIME_REPORTING.csv"
SUMMARY_FILE = r"C:\Spring 2026\American Airlines Challenge\Data\BTS Yearly\2025 Jan-Dec\ot_delaycause1_DL (3)\Airline_Delay_Cause.csv"
# ────────────────────────────────────────────────────────────────────────────

# ── STEP 1: Load the flight-level data ──────────────────────────────────────
print("Loading flight data...")
flights = pd.read_csv(FLIGHT_FILE, low_memory=False)
print(f"Total rows loaded: {len(flights)}")
print("Columns:", flights.columns.tolist())

# ── STEP 2: Filter to American Airlines only ────────────────────────────────
flights = flights[flights["OP_UNIQUE_CARRIER"] == "AA"]
print(f"Rows after filtering to AA: {len(flights)}")

# ── STEP 3: Separate inbound and outbound legs at DFW ───────────────────────
inbound = flights[flights["DEST"] == "DFW"].copy()
inbound = inbound.rename(columns={
    "ORIGIN":         "airport_A",
    "ARR_DELAY":      "inbound_arr_delay",
    "DEP_DELAY":      "inbound_dep_delay",
    "WEATHER_DELAY":  "inbound_weather_delay",
    "NAS_DELAY":      "inbound_nas_delay",
    "CRS_ARR_TIME":   "inbound_sched_arr",
    "ARR_TIME":       "inbound_actual_arr",
    "CANCELLED":      "inbound_cancelled"
})
print(f"Inbound legs (A→DFW): {len(inbound)}")

outbound = flights[flights["ORIGIN"] == "DFW"].copy()
outbound = outbound.rename(columns={
    "DEST":           "airport_B",
    "DEP_DELAY":      "outbound_dep_delay",
    "WEATHER_DELAY":  "outbound_weather_delay",
    "NAS_DELAY":      "outbound_nas_delay",
    "CRS_DEP_TIME":   "outbound_sched_dep",
    "DEP_TIME":       "outbound_actual_dep",
    "CANCELLED":      "outbound_cancelled"
})
print(f"Outbound legs (DFW→B): {len(outbound)}")

# ── STEP 4: Link inbound and outbound via Tail Number + Date ────────────────
inbound_cols  = ["FL_DATE", "TAIL_NUM", "airport_A",
                 "inbound_arr_delay", "inbound_weather_delay",
                 "inbound_nas_delay", "inbound_sched_arr",
                 "inbound_actual_arr", "inbound_cancelled"]

outbound_cols = ["FL_DATE", "TAIL_NUM", "airport_B",
                 "outbound_dep_delay", "outbound_weather_delay",
                 "outbound_nas_delay", "outbound_sched_dep",
                 "outbound_actual_dep", "outbound_cancelled"]

sequences = pd.merge(
    inbound[inbound_cols],
    outbound[outbound_cols],
    on=["FL_DATE", "TAIL_NUM"],
    how="inner"
)
print(f"Matched A→DFW→B sequences: {len(sequences)}")

# ── STEP 5: Compute connection time ─────────────────────────────────────────
sequences["connection_minutes"] = (
    sequences["outbound_sched_dep"].astype(float) -
    sequences["inbound_sched_arr"].astype(float)
)
sequences = sequences[sequences["connection_minutes"] > 0]
print(f"Sequences with valid connection time: {len(sequences)}")

# ── STEP 6: Create cascading delay label (FIXED) ────────────────────────────

# FIX 1: Capture ALL delay types, not just weather and NAS
sequences["inbound_disrupted"] = (
    (sequences["inbound_weather_delay"].fillna(0) +
     sequences["inbound_nas_delay"].fillna(0) +
     sequences["inbound_arr_delay"].fillna(0)) >= 30
).astype(int)

sequences["outbound_disrupted"] = (
    sequences["outbound_dep_delay"].fillna(0) >= 20
).astype(int)

# FIX 2: Treat cancelled outbound as a cascading event too
sequences["CASCADING_DELAY"] = (
    ((sequences["inbound_disrupted"] == 1) &
     (sequences["outbound_disrupted"] == 1)) |
    ((sequences["inbound_disrupted"] == 1) &
     (sequences["outbound_cancelled"] == 1.0))
).astype(int)

print(f"\nCascading delay breakdown:")
print(sequences["CASCADING_DELAY"].value_counts())
print(f"Cascade rate: {sequences['CASCADING_DELAY'].mean():.2%}")

# ── STEP 7: Load and merge airport summary features ─────────────────────────
print("\nLoading airport summary data...")
summary = pd.read_csv(SUMMARY_FILE, low_memory=False)
print("Summary columns:", summary.columns.tolist())

summary["weather_delay_rate"] = (
    summary["weather_ct"] / summary["arr_flights"].replace(0, np.nan)
)
summary["nas_delay_rate"] = (
    summary["nas_ct"] / summary["arr_flights"].replace(0, np.nan)
)

summary_slim = summary[["airport", "month", "weather_delay_rate", "nas_delay_rate"]].copy()

sequences["month"] = pd.to_datetime(sequences["FL_DATE"]).dt.month

# Merge airport A features
sequences = sequences.merge(
    summary_slim.rename(columns={
        "airport": "airport_A",
        "weather_delay_rate": "A_weather_delay_rate",
        "nas_delay_rate": "A_nas_delay_rate"
    }),
    on=["airport_A", "month"], how="left"
)

# Merge airport B features
sequences = sequences.merge(
    summary_slim.rename(columns={
        "airport": "airport_B",
        "weather_delay_rate": "B_weather_delay_rate",
        "nas_delay_rate": "B_nas_delay_rate"
    }),
    on=["airport_B", "month"], how="left"
)

print(f"\nFinal dataset shape: {sequences.shape}")
print(sequences.head())

# ── STEP 8: Save output ──────────────────────────────────────────────────────
sequences.to_csv("DFW_sequences_Jan2025.csv", index=False)
print("\nSaved to DFW_sequences_Jan2025.csv")