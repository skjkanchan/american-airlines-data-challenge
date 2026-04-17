import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from itertools import product

# ── FILE PATHS ───────────────────────────────────────────────────────────────
MODEL_FILE      = r"C:\Spring 2026\american-airlines-data-challenge\embedding_model.pth"
ENCODER_FILE    = r"C:\Spring 2026\american-airlines-data-challenge\airport_label_encoder.npy"
SEQUENCES_FILE  = r"C:\Spring 2026\american-airlines-data-challenge\DFW_sequences_enriched.csv"
OUTPUT_FILE     = r"C:\Spring 2026\american-airlines-data-challenge\risky_pairings.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── STEP 1: Rebuild model architecture (must match training script) ──────────
class AirportEmbeddingModel(nn.Module):
    def __init__(self, num_airports, embedding_dim=32, num_extra_features=42):
        super(AirportEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_airports, embedding_dim)
        input_dim = embedding_dim * 2 + num_extra_features
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

# ── STEP 2: Load model and encoder ──────────────────────────────────────────
print("Loading model and encoder...")
classes      = np.load(ENCODER_FILE, allow_pickle=True)
le           = LabelEncoder()
le.classes_  = classes
num_airports = len(classes)

model = AirportEmbeddingModel(num_airports=num_airports)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()
print(f"Model loaded — {num_airports} airports")

# ── STEP 3: Load enriched sequences to compute per-airport feature averages ──
# We score each pairing using the average feature profile for each airport
# This represents a "typical" sequence for that airport under average conditions
print("Loading enriched sequences...")
df = pd.read_csv(SEQUENCES_FILE, low_memory=False)

feature_cols = [
    "connection_minutes", "month_sin", "month_cos",
    "A_weather_delay_rate", "A_nas_delay_rate",
    "B_weather_delay_rate", "B_nas_delay_rate",
    "pair_simultaneous_weather_risk",
    "A_avg_delay_rate", "A_avg_delay_minutes",
    "A_delay_volatility", "A_gdp_proxy_score",
    "B_avg_delay_rate", "B_avg_delay_minutes",
    "B_delay_volatility", "B_gdp_proxy_score",
    "A_visibility", "A_wind_speed", "A_storm",
    "A_ifr", "A_severity", "A_bad_weather",
    "B_visibility", "B_wind_speed", "B_storm",
    "B_ifr", "B_severity", "B_bad_weather",
    "leg_number_in_duty", "elapsed_duty_hours",
    "rest_hours_before_duty", "block_hours_last_24h",
    "block_hours_last_7d", "remaining_duty_hours",
    "duty_violation_risk",
    "circadian_risk_flag", "early_morning_flag",
    "redeye_flag", "sequence_complexity",
    "below_mct_flag", "connection_buffer_ratio",
    "adjusted_connection_risk"
]

# Compute per-airport average feature profiles
print("Computing airport feature profiles...")

# Features that belong to Airport A side
a_features = [
    "A_weather_delay_rate", "A_nas_delay_rate",
    "A_avg_delay_rate", "A_avg_delay_minutes",
    "A_delay_volatility", "A_gdp_proxy_score",
    "A_visibility", "A_wind_speed", "A_storm",
    "A_ifr", "A_severity", "A_bad_weather"
]

# Features that belong to Airport B side
b_features = [
    "B_weather_delay_rate", "B_nas_delay_rate",
    "B_avg_delay_rate", "B_avg_delay_minutes",
    "B_delay_volatility", "B_gdp_proxy_score",
    "B_visibility", "B_wind_speed", "B_storm",
    "B_ifr", "B_severity", "B_bad_weather"
]

# Average A-side profile per airport
airport_a_profile = df.groupby("airport_A")[a_features].mean()

# Average B-side profile per airport
airport_b_profile = df.groupby("airport_B")[b_features].mean()

# Global averages for sequence-level features (used for all pairs)
global_avg = df[feature_cols].mean()

# ── STEP 4: Score all airport pairs ─────────────────────────────────────────
print(f"Scoring all {num_airports}x{num_airports} airport pairs...")
print(f"Total combinations: {num_airports * num_airports:,}")

results = []
airports_list = list(classes)

# Process in batches for efficiency
BATCH_SIZE = 512
pairs = [(a, b) for a, b in product(airports_list, airports_list) if a != b]
print(f"Pairs to score (excluding A=B): {len(pairs):,}")

for i in range(0, len(pairs), BATCH_SIZE):
    batch = pairs[i:i + BATCH_SIZE]

    airport_a_ids = []
    airport_b_ids = []
    feature_vectors = []

    for airport_a, airport_b in batch:
        # Get integer IDs
        a_id = le.transform([airport_a])[0]
        b_id = le.transform([airport_b])[0]
        airport_a_ids.append(a_id)
        airport_b_ids.append(b_id)

        # Build feature vector using per-airport averages
        fv = global_avg.copy()

        # Override with airport-specific averages where available
        if airport_a in airport_a_profile.index:
            for col in a_features:
                fv[col] = airport_a_profile.loc[airport_a, col]

        if airport_b in airport_b_profile.index:
            for col in b_features:
                fv[col] = airport_b_profile.loc[airport_b, col]

        # Pair-level simultaneous weather risk
        fv["pair_simultaneous_weather_risk"] = (
            fv["A_weather_delay_rate"] * fv["B_weather_delay_rate"]
        )

        # Adjusted connection risk
        fv["adjusted_connection_risk"] = (
            fv["below_mct_flag"] * (1 + fv["A_nas_delay_rate"])
        )

        feature_vectors.append(fv[feature_cols].values.astype(float))

    # Convert to tensors
    a_tensor = torch.tensor(airport_a_ids, dtype=torch.long)
    b_tensor = torch.tensor(airport_b_ids, dtype=torch.long)
    f_tensor = torch.tensor(np.array(feature_vectors), dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        probs = model(a_tensor, b_tensor, f_tensor).numpy()

    for (airport_a, airport_b), prob in zip(batch, probs):
        results.append({
            "airport_A":           airport_a,
            "airport_B":           airport_b,
            "cascade_probability": round(float(prob), 4)
        })

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Scored {min(i + BATCH_SIZE, len(pairs)):,} / {len(pairs):,} pairs...")

# ── STEP 5: Build output dataframe ──────────────────────────────────────────
print("\nBuilding results...")
results_df = pd.DataFrame(results)

# Add risk level labels
def risk_label(p):
    if p >= 0.60:   return "CRITICAL"
    elif p >= 0.45: return "HIGH"
    elif p >= 0.30: return "ELEVATED"
    else:           return "NORMAL"

results_df["risk_level"] = results_df["cascade_probability"].apply(risk_label)

# Sort by cascade probability descending
results_df = results_df.sort_values("cascade_probability", ascending=False).reset_index(drop=True)
results_df.insert(0, "rank", results_df.index + 1)

# Add sequence label for readability
results_df["sequence"] = (
    results_df["airport_A"] + " → DFW → " + results_df["airport_B"]
)

# Reorder columns
results_df = results_df[[
    "rank", "sequence", "airport_A", "airport_B",
    "cascade_probability", "risk_level"
]]

# ── STEP 6: Save and print summary ──────────────────────────────────────────
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(results_df):,} scored pairings to {OUTPUT_FILE}")

print("\n" + "="*60)
print("RISK LEVEL SUMMARY")
print("="*60)
print(results_df["risk_level"].value_counts().to_string())

print("\n" + "="*60)
print("TOP 25 HIGHEST RISK PAIRINGS TO AVOID")
print("="*60)
print(results_df.head(25)[["rank","sequence","cascade_probability","risk_level"]].to_string(index=False))

print("\n" + "="*60)
print("TOP 10 SAFEST PAIRINGS")
print("="*60)
print(results_df.tail(10)[["rank","sequence","cascade_probability","risk_level"]].to_string(index=False))

# ── STEP 7: Airport-level avoid summary ─────────────────────────────────────
print("\n" + "="*60)
print("TOP 15 AIRPORTS WITH MOST HIGH-RISK OUTBOUND PAIRINGS")
print("="*60)
high_risk = results_df[results_df["risk_level"].isin(["CRITICAL","HIGH"])]
outbound_count = high_risk.groupby("airport_A").size().sort_values(ascending=False)
print(outbound_count.head(15).to_string())

print("\n" + "="*60)
print("TOP 15 AIRPORTS WITH MOST HIGH-RISK INBOUND PAIRINGS")
print("="*60)
inbound_count = high_risk.groupby("airport_B").size().sort_values(ascending=False)
print(inbound_count.head(15).to_string())