import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── FILE PATHS ───────────────────────────────────────────────────────────────
MONTHLY_DATA_DIR = r"C:\Spring 2026\american-airlines-data-challenge\Data\Monthly Data"
WEATHER_FILE      = r"C:\Spring 2026\american-airlines-data-challenge\Data\weather-AWC\processed\weather_all_airports_clean.csv"
AIRPORT_FEATURES  = r"C:\Spring 2026\american-airlines-data-challenge\airport_features.csv"
OUTPUT_SEQUENCES  = r"C:\Spring 2026\american-airlines-data-challenge\DFW_sequences_enriched.csv"
MODEL_SAVE        = r"C:\Spring 2026\american-airlines-data-challenge\embedding_model.pth"
ENCODER_SAVE      = r"C:\Spring 2026\american-airlines-data-challenge\airport_label_encoder.npy"

# ── STEP 1: Load and combine all 12 monthly BTS CSVs ────────────────────────
print("Loading BTS monthly data...")
all_files = glob.glob(f"{MONTHLY_DATA_DIR}\\*.csv")
print(f"Files found: {all_files}")  # add this to confirm
print(f"Found {len(all_files)} monthly files")

bts = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in all_files],
    ignore_index=True
)
print(f"Total BTS rows loaded: {len(bts)}")

# Filter to American Airlines + DFW only
bts = bts[
    (bts["OP_UNIQUE_CARRIER"] == "AA") &
    (bts["ORIGIN"].eq("DFW") | bts["DEST"].eq("DFW"))
].copy()
print(f"Rows after AA + DFW filter: {len(bts)}")

# Drop cancelled flights
bts = bts[bts["CANCELLED"] != 1].copy()

# Parse date and month
bts["FL_DATE"] = pd.to_datetime(bts["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
bts["month"]   = bts["FL_DATE"].dt.month

# Build departure and arrival datetimes (rounded to hour for METAR join)
def build_datetime(date_series, time_series):
    hour_str = time_series.astype(str).str.zfill(4).str[:2]
    return pd.to_datetime(
        date_series.astype(str) + " " + hour_str + ":00",
        errors="coerce"
    )

bts["dep_datetime"] = build_datetime(bts["FL_DATE"], bts["CRS_DEP_TIME"])
bts["arr_datetime"] = build_datetime(bts["FL_DATE"], bts["CRS_ARR_TIME"])
bts["block_hours"]  = (
    (bts["arr_datetime"] - bts["dep_datetime"])
    .dt.total_seconds() / 3600
).clip(0, 20)

# ── STEP 2: Build A→DFW→B sequences ─────────────────────────────────────────
print("\nBuilding A→DFW→B sequences...")

inbound = bts[bts["DEST"] == "DFW"][[
    "TAIL_NUM", "FL_DATE", "month", "ORIGIN",
    "CRS_ARR_TIME", "ARR_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "dep_datetime", "arr_datetime"
]].rename(columns={
    "ORIGIN":        "airport_A",
    "CRS_ARR_TIME":  "A_crs_arr",
    "ARR_DELAY":     "A_arr_delay",
    "WEATHER_DELAY": "A_weather_delay_mins",
    "NAS_DELAY":     "A_nas_delay_mins",
    "dep_datetime":  "A_dep_datetime",
    "arr_datetime":  "A_arr_datetime"
})

outbound = bts[bts["ORIGIN"] == "DFW"][[
    "TAIL_NUM", "FL_DATE", "DEST",
    "CRS_DEP_TIME", "DEP_DELAY", "ARR_DELAY",
    "dep_datetime"
]].rename(columns={
    "DEST":         "airport_B",
    "CRS_DEP_TIME": "B_crs_dep",
    "DEP_DELAY":    "B_dep_delay",
    "ARR_DELAY":    "B_arr_delay",
    "dep_datetime": "B_dep_datetime"
})

# Join on same tail + same date
sequences = inbound.merge(outbound, on=["TAIL_NUM", "FL_DATE"])

# Connection time in minutes
sequences["connection_minutes"] = sequences["B_crs_dep"] - sequences["A_crs_arr"]
sequences.loc[sequences["connection_minutes"] < 0, "connection_minutes"] += 2400

# Keep realistic connections only (20 min to 4 hours)
sequences = sequences[
    sequences["connection_minutes"].between(20, 240)
].copy()

# Cascading delay label
sequences["CASCADING_DELAY"] = (
    (sequences["A_arr_delay"]  > 15) &
    (sequences["B_dep_delay"]  > 15) &
    (sequences["B_dep_delay"] >= sequences["A_arr_delay"] * 0.5)
).astype(int)

print(f"Sequences built: {len(sequences)}")
print(f"Cascade rate:    {sequences['CASCADING_DELAY'].mean():.2%}")

# ── STEP 3: Join airport features (ASPM) ────────────────────────────────────
print("\nJoining airport features...")
aspm = pd.read_csv(AIRPORT_FEATURES)
# Columns: airport, avg_delay_rate, avg_delay_minutes, delay_volatility, gdp_proxy_score

for side in ["A", "B"]:
    sequences = sequences.merge(
        aspm.rename(columns={
            "airport":         f"airport_{side}",
            "avg_delay_rate":  f"{side}_avg_delay_rate",
            "avg_delay_minutes": f"{side}_avg_delay_minutes",
            "delay_volatility":  f"{side}_delay_volatility",
            "gdp_proxy_score":   f"{side}_gdp_proxy_score"
        }),
        on=f"airport_{side}", how="left"
    )

# ── STEP 4: BTS-derived weather rates ───────────────────────────────────────
print("Computing BTS weather rates...")

weather_rates = bts.groupby("ORIGIN").agg(
    total_flights        = ("FL_DATE", "count"),
    weather_delay_flights = ("WEATHER_DELAY", lambda x: (x.fillna(0) > 0).sum()),
    nas_delay_flights    = ("NAS_DELAY",      lambda x: (x.fillna(0) > 0).sum())
).reset_index()
weather_rates["weather_delay_rate"] = (
    weather_rates["weather_delay_flights"] / weather_rates["total_flights"]
)
weather_rates["nas_delay_rate"] = (
    weather_rates["nas_delay_flights"] / weather_rates["total_flights"]
)

for side in ["A", "B"]:
    sequences = sequences.merge(
        weather_rates[["ORIGIN", "weather_delay_rate", "nas_delay_rate"]].rename(columns={
            "ORIGIN":             f"airport_{side}",
            "weather_delay_rate": f"{side}_weather_delay_rate",
            "nas_delay_rate":     f"{side}_nas_delay_rate"
        }),
        on=f"airport_{side}", how="left"
    )

sequences["pair_simultaneous_weather_risk"] = (
    sequences["A_weather_delay_rate"] * sequences["B_weather_delay_rate"]
)

# ── STEP 5: Join METAR weather features ─────────────────────────────────────
print("Joining METAR weather features...")
metar = pd.read_csv(
    r"C:\Spring 2026\american-airlines-data-challenge\Data\weather-NOAA\metar_2025.csv",
    low_memory=False
)

# Parse datetime and ROUND to nearest hour (METAR obs are at :52, rounding aligns them)
metar["valid"] = pd.to_datetime(metar["valid"], errors="coerce")
metar["join_hour"] = metar["valid"].dt.round("h")

# Clean visibility — replace 'M' (missing) with 10 (clear)
metar["vsby"] = pd.to_numeric(
    metar["vsby"].replace("M", None), errors="coerce"
).fillna(10)

# Clean wind speed
metar["sknt"] = pd.to_numeric(
    metar["sknt"].replace("M", None), errors="coerce"
).fillna(0)

# Ceiling height — skyl1 is in feet, M means missing/clear
metar["skyl1"] = pd.to_numeric(
    metar["skyl1"].replace("M", None), errors="coerce"
).fillna(10000)

# Ceiling flag — BKN or OVC means actual ceiling exists
metar["ceiling_ft"] = metar.apply(
    lambda r: r["skyl1"] if r["skyc1"] in ["BKN", "OVC"] else 10000, axis=1
)

# IFR conditions: ceiling < 1000ft OR visibility < 3sm
metar["ifr"] = (
    (metar["ceiling_ft"] < 1000) | (metar["vsby"] < 3)
).astype(int)

# Storm flag
metar["storm"] = metar["wxcodes"].apply(
    lambda x: 1 if isinstance(x, str) and
    any(code in x for code in ["TS", "RA", "SN", "FZ", "GR", "PL"]) else 0
)

# Severity score
metar["severity"] = (
    metar["storm"] * 2 +
    metar["ifr"] * 2 +
    (metar["sknt"] > 25).astype(int)
).clip(0, 5)

# Bad weather flag
metar["bad_weather"] = (metar["severity"] >= 2).astype(int)

metar_clean = metar[[
    "station", "join_hour",
    "vsby", "sknt", "storm", "ifr", "severity", "bad_weather"
]].dropna(subset=["station", "join_hour"])

# ── TIMEZONE OFFSET: convert BTS local time to UTC for METAR join ────────────
timezone_offsets = {
    'ABQ':5,'AGS':5,'ALB':5,'AMA':6,'ANC':9,'ATL':5,'AUS':6,'AVL':5,
    'BDL':5,'BFL':8,'BHM':6,'BNA':6,'BOI':7,'BOS':5,'BUF':5,'BUR':8,
    'BWI':5,'BZN':7,'CAE':5,'CHS':5,'CID':6,'CLE':5,'CLT':5,'CMH':5,
    'COS':7,'CVG':5,'DAB':5,'DCA':5,'DEN':7,'DRO':7,'DSM':6,'DTW':5,
    'ECP':6,'EGE':7,'ELP':7,'EUG':8,'EWR':5,'EYW':5,'FAT':8,'FCA':7,
    'FLL':5,'FSD':6,'GEG':8,'GRR':5,'GSO':5,'GSP':5,'GUC':7,'HDN':7,
    'HNL':10,'HSV':6,'IAD':5,'IAH':6,'ICT':6,'ILM':5,'IND':5,'JAC':7,
    'JAX':5,'JFK':5,'KOA':10,'LAS':8,'LAX':8,'LBB':6,'LEX':5,'LGA':5,
    'LIT':6,'MCI':6,'MCO':5,'MDT':5,'MEM':6,'MFE':6,'MIA':5,'MKE':6,
    'MRY':8,'MSN':6,'MSO':7,'MSP':6,'MSY':6,'MTJ':7,'MYR':5,'OGG':10,
    'OKC':6,'OMA':6,'ONT':8,'ORD':6,'ORF':5,'PBI':5,'PDX':8,'PHL':5,
    'PHX':7,'PIT':5,'PNS':6,'PSP':8,'PWM':5,'RDM':8,'RDU':5,'RIC':5,
    'RNO':8,'RSW':5,'SAN':8,'SAT':6,'SAV':5,'SBA':8,'SBP':8,'SDF':5,
    'SEA':8,'SFO':8,'SJC':8,'SJU':4,'SLC':7,'SMF':8,'SNA':8,'SRQ':5,
    'STL':6,'STS':8,'STT':4,'SYR':5,'TPA':5,'TUL':6,'TUS':7,'TVC':5,
    'TYS':5,'VPS':6,'XNA':6
}

def local_to_utc(dt_series, airport_series):
    offsets = airport_series.map(timezone_offsets).fillna(6)
    return dt_series + pd.to_timedelta(offsets, unit='h')

# Compute UTC join hours — rounded to nearest hour
sequences["A_join_hour"] = local_to_utc(
    sequences["A_dep_datetime"], sequences["airport_A"]
).dt.round("h")

sequences["B_join_hour"] = local_to_utc(
    sequences["B_dep_datetime"], sequences["airport_B"]
).dt.round("h")

# Join METAR for Airport A
sequences = sequences.merge(
    metar_clean.rename(columns={
        "station":     "airport_A",
        "join_hour":   "A_join_hour",
        "vsby":        "A_visibility",
        "sknt":        "A_wind_speed",
        "storm":       "A_storm",
        "ifr":         "A_ifr",
        "severity":    "A_severity",
        "bad_weather": "A_bad_weather"
    }),
    on=["airport_A", "A_join_hour"], how="left"
)

# Join METAR for Airport B
sequences = sequences.merge(
    metar_clean.rename(columns={
        "station":     "airport_B",
        "join_hour":   "B_join_hour",
        "vsby":        "B_visibility",
        "sknt":        "B_wind_speed",
        "storm":       "B_storm",
        "ifr":         "B_ifr",
        "severity":    "B_severity",
        "bad_weather": "B_bad_weather"
    }),
    on=["airport_B", "B_join_hour"], how="left"
)

# Fill missing METAR with safe defaults (clear conditions)
metar_defaults = {
    "A_visibility": 10, "A_wind_speed": 0, "A_storm": 0,
    "A_ifr": 0, "A_severity": 0, "A_bad_weather": 0,
    "B_visibility": 10, "B_wind_speed": 0, "B_storm": 0,
    "B_ifr": 0, "B_severity": 0, "B_bad_weather": 0
}
for col, val in metar_defaults.items():
    sequences[col] = sequences[col].fillna(val)

# Verify join quality
a_fill = (sequences["A_visibility"] == 10).mean()
b_fill = (sequences["B_visibility"] == 10).mean()
print(f"A weather fill rate (lower is better): {a_fill:.2%}")
print(f"B weather fill rate (lower is better): {b_fill:.2%}")

# ── STEP 6: Duty time features ───────────────────────────────────────────────
print("Computing duty time features (this may take a few minutes)...")

FAR117_REST = 10.0
bts_sorted  = bts.sort_values(["TAIL_NUM", "dep_datetime"]).copy()

duty_records = []
for tail, group in bts_sorted.groupby("TAIL_NUM"):
    group       = group.reset_index(drop=True)
    duty_start  = None
    last_arr    = None
    leg_in_duty = 0
    history     = []  # list of (dep_time, block_hours)

    for _, row in group.iterrows():
        dep = row["dep_datetime"]
        arr = row["arr_datetime"]
        if pd.isnull(dep) or pd.isnull(arr):
            continue

        rest_hours = (
            (dep - last_arr).total_seconds() / 3600
            if last_arr else 12.0
        )

        if last_arr is None or rest_hours >= FAR117_REST:
            duty_start  = dep
            leg_in_duty = 1
        else:
            leg_in_duty += 1

        elapsed_duty   = (dep - duty_start).total_seconds() / 3600
        remaining_duty = max(0, 9.0 - elapsed_duty)

        # Rolling block hour windows
        history = [(t, h) for t, h in history
                   if (dep - t).total_seconds() / 3600 <= 168]
        bh_24 = sum(h for t, h in history
                    if (dep - t).total_seconds() / 3600 <= 24)
        bh_7d = sum(h for _, h in history)

        violation = int(
            elapsed_duty > 9.0 * 0.85 or
            rest_hours   < FAR117_REST  or
            bh_24        > 8.0  * 0.9  or
            bh_7d        > 30.0 * 0.9
        )

        duty_records.append({
            "TAIL_NUM":             tail,
            "dep_datetime_key":     dep.floor("min"),
            "leg_number_in_duty":   leg_in_duty,
            "elapsed_duty_hours":   round(elapsed_duty, 3),
            "rest_hours_before_duty": round(rest_hours, 3),
            "block_hours_last_24h": round(bh_24, 3),
            "block_hours_last_7d":  round(bh_7d, 3),
            "remaining_duty_hours": round(remaining_duty, 3),
            "duty_violation_risk":  violation
        })

        history.append((dep, row["block_hours"]))
        last_arr = arr

duty_df = pd.DataFrame(duty_records)

sequences["dep_datetime_key"] = sequences["A_dep_datetime"].dt.floor("min")
sequences = sequences.merge(
    duty_df, on=["TAIL_NUM", "dep_datetime_key"], how="left"
)

duty_defaults = {
    "leg_number_in_duty":     1,
    "elapsed_duty_hours":     0,
    "rest_hours_before_duty": 12,
    "block_hours_last_24h":   0,
    "block_hours_last_7d":    0,
    "remaining_duty_hours":   9,
    "duty_violation_risk":    0
}
for col, val in duty_defaults.items():
    sequences[col] = sequences[col].fillna(val)

# ── STEP 7: Fatigue features ─────────────────────────────────────────────────
sequences["dep_hour"] = sequences["A_dep_datetime"].dt.hour

sequences["circadian_risk_flag"] = (
    sequences["dep_hour"].between(2, 5)
).astype(float)
sequences["early_morning_flag"] = (
    sequences["dep_hour"] < 6
).astype(float)
sequences["redeye_flag"] = (
    (sequences["dep_hour"] >= 22) | (sequences["dep_hour"] <= 2)
).astype(float)
sequences["sequence_complexity"] = (
    sequences["leg_number_in_duty"].clip(upper=6)
)

# ── STEP 8: Seasonality encoding ─────────────────────────────────────────────
sequences["month_sin"] = np.sin(2 * np.pi * sequences["month"] / 12)
sequences["month_cos"] = np.cos(2 * np.pi * sequences["month"] / 12)

# ── STEP 9: Connection risk features ────────────────────────────────────────
DFW_MCT = 30

sequences["below_mct_flag"] = (
    sequences["connection_minutes"] < DFW_MCT
).astype(float)
sequences["connection_buffer_ratio"] = (
    (sequences["connection_minutes"] - DFW_MCT) / DFW_MCT
).clip(-1.0, 5.0)
sequences["adjusted_connection_risk"] = (
    sequences["below_mct_flag"] * (1 + sequences["A_nas_delay_rate"].fillna(0))
)

# ── STEP 10: Define feature columns ─────────────────────────────────────────
feature_cols = [
    # Seasonality
    "connection_minutes", "month_sin", "month_cos",

    # BTS weather rates
    "A_weather_delay_rate", "A_nas_delay_rate",
    "B_weather_delay_rate", "B_nas_delay_rate",
    "pair_simultaneous_weather_risk",

    # ASPM airport stats
    "A_avg_delay_rate", "A_avg_delay_minutes",
    "A_delay_volatility", "A_gdp_proxy_score",
    "B_avg_delay_rate", "B_avg_delay_minutes",
    "B_delay_volatility", "B_gdp_proxy_score",

    # METAR weather
    "A_visibility", "A_wind_speed", "A_storm",
    "A_ifr", "A_severity", "A_bad_weather",
    "B_visibility", "B_wind_speed", "B_storm",
    "B_ifr", "B_severity", "B_bad_weather",

    # Duty time
    "leg_number_in_duty", "elapsed_duty_hours",
    "rest_hours_before_duty", "block_hours_last_24h",
    "block_hours_last_7d", "remaining_duty_hours",
    "duty_violation_risk",

    # Fatigue
    "circadian_risk_flag", "early_morning_flag",
    "redeye_flag", "sequence_complexity",

    # Connection risk
    "below_mct_flag", "connection_buffer_ratio",
    "adjusted_connection_risk"
]

print(f"\nTotal features: {len(feature_cols)}")

# Fill any remaining NaNs
sequences[feature_cols] = sequences[feature_cols].fillna(0)

# Verify clean
null_check = sequences[feature_cols].isnull().sum()
if null_check.sum() > 0:
    print("WARNING — NaNs remain:")
    print(null_check[null_check > 0])
else:
    print("All feature columns clean — no NaNs.")

# Save enriched sequences
sequences.to_csv(OUTPUT_SEQUENCES, index=False)
print(f"Saved enriched sequences to {OUTPUT_SEQUENCES}")
print(f"Final sequence count: {len(sequences)}")
print(f"Cascade rate: {sequences['CASCADING_DELAY'].mean():.2%}")

# ── STEP 11: Encode airports ─────────────────────────────────────────────────
print("\nEncoding airports...")
le          = LabelEncoder()
all_airports = pd.concat([sequences["airport_A"], sequences["airport_B"]]).unique()
le.fit(all_airports)

sequences["airport_A_id"] = le.transform(sequences["airport_A"])
sequences["airport_B_id"] = le.transform(sequences["airport_B"])
num_airports = len(le.classes_)
print(f"Unique airports: {num_airports}")

# ── STEP 12: Train/test split ────────────────────────────────────────────────
train_df, test_df = train_test_split(
    sequences, test_size=0.2, random_state=42,
    stratify=sequences["CASCADING_DELAY"]
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── STEP 13: PyTorch Dataset ─────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.airport_a = torch.tensor(dataframe["airport_A_id"].values, dtype=torch.long)
        self.airport_b = torch.tensor(dataframe["airport_B_id"].values, dtype=torch.long)
        self.features  = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
        self.labels    = torch.tensor(dataframe["CASCADING_DELAY"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.airport_a[idx], self.airport_b[idx], self.features[idx], self.labels[idx]

train_dataset = SequenceDataset(train_df)
test_dataset  = SequenceDataset(test_df)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ── STEP 14: Model architecture ──────────────────────────────────────────────
class AirportEmbeddingModel(nn.Module):
    def __init__(self, num_airports, embedding_dim=32, num_extra_features=43):
        super(AirportEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_airports, embedding_dim)

        input_dim = embedding_dim * 2 + num_extra_features  # 32+32+43 = 107

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, airport_a, airport_b, features):
        emb_a = self.embedding(airport_a)
        emb_b = self.embedding(airport_b)
        x = torch.cat([emb_a, emb_b, features], dim=1)
        return self.network(x).squeeze(1)

num_features = len(feature_cols)
model = AirportEmbeddingModel(
    num_airports=num_airports,
    num_extra_features=num_features
)
print(f"\nModel input dimension: 64 (embeddings) + {num_features} (features) = {64 + num_features}")
print(model)

# ── STEP 15: Class imbalance ─────────────────────────────────────────────────
n_neg      = (sequences["CASCADING_DELAY"] == 0).sum()
n_pos      = (sequences["CASCADING_DELAY"] == 1).sum()
pos_weight = torch.tensor([3.0], dtype=torch.float32)
print(f"\nNo cascade: {n_neg} | Cascade: {n_pos}")
print(f"Positive class weight: {pos_weight.item():.2f}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ── STEP 16: Training loop ───────────────────────────────────────────────────
EPOCHS      = 50
train_losses = []

print("\nTraining...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for airport_a, airport_b, features, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(airport_a, airport_b, features)

        weights = torch.where(
            labels == 1,
            pos_weight.expand_as(labels),
            torch.ones_like(labels)
        )
        loss = nn.functional.binary_cross_entropy(predictions, labels, weight=weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# ── STEP 17: Evaluation ──────────────────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for airport_a, airport_b, features, labels in test_loader:
        preds = model(airport_a, airport_b, features)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds    = np.array(all_preds)
all_labels   = np.array(all_labels)
binary_preds = (all_preds >= 0.3).astype(int)

print("\nClassification Report:")
print(classification_report(all_labels, binary_preds,
                             target_names=["No Cascade", "Cascade"]))
print(f"ROC-AUC Score:        {roc_auc_score(all_labels, all_preds):.4f}")
print(f"Precision-Recall AUC: {average_precision_score(all_labels, all_preds):.4f}")

# ── STEP 18: Training loss plot ──────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Spring 2026\american-airlines-data-challenge\training_loss.png")
plt.show()
print("Training loss plot saved.")

# ── STEP 19: Save model and encoder ─────────────────────────────────────────
torch.save(model.state_dict(), MODEL_SAVE)
np.save(ENCODER_SAVE, le.classes_)
print(f"Model saved to {MODEL_SAVE}")
print(f"Encoder saved to {ENCODER_SAVE}")