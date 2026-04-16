import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── FILE PATHS ───────────────────────────────────────────────────────────────
DATA_FILE    = r"C:\Spring 2026\American Airlines Challenge\scripts\DFW_sequences_Jan2025.csv"
MODEL_FILE   = r"C:\Spring 2026\American Airlines Challenge\scripts\embedding_model.pth"
ENCODER_FILE = r"C:\Spring 2026\American Airlines Challenge\scripts\airport_label_encoder.npy"
# ────────────────────────────────────────────────────────────────────────────

# ── STEP 1: Rebuild the model architecture (must match training script) ──────
class AirportEmbeddingModel(nn.Module):
    def __init__(self, num_airports, embedding_dim=32, num_extra_features=7):
        super(AirportEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_airports, embedding_dim)
        input_dim = embedding_dim * 2 + num_extra_features
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),          nn.Sigmoid()
        )

    def forward(self, airport_a, airport_b, features):
        emb_a = self.embedding(airport_a)
        emb_b = self.embedding(airport_b)
        x = torch.cat([emb_a, emb_b, features], dim=1)
        return self.network(x).squeeze(1)

# ── STEP 2: Load label encoder and model ────────────────────────────────────
print("Loading model and encoder...")
classes     = np.load(ENCODER_FILE, allow_pickle=True)
le          = LabelEncoder()
le.classes_ = classes
num_airports = len(classes)

model = AirportEmbeddingModel(num_airports=num_airports)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()
print(f"Model loaded — {num_airports} airports")

# ── STEP 3: Extract embeddings for all airports ──────────────────────────────
print("Extracting embeddings...")
all_ids = torch.arange(num_airports, dtype=torch.long)
with torch.no_grad():
    embeddings = model.embedding(all_ids).numpy()  # shape: (107, 32)
print(f"Embedding matrix shape: {embeddings.shape}")

# ── STEP 4: Load data to compute per-airport cascade rate ────────────────────
print("Computing airport risk scores...")
df = pd.read_csv(DATA_FILE, low_memory=False)

# Cascade rate as Airport A (inbound)
risk_A = df.groupby("airport_A")["CASCADING_DELAY"].mean().rename("risk_as_A")

# Cascade rate as Airport B (outbound)
risk_B = df.groupby("airport_B")["CASCADING_DELAY"].mean().rename("risk_as_B")

# Combined average risk
risk = pd.concat([risk_A, risk_B], axis=1).fillna(0)
risk["avg_risk"] = (risk["risk_as_A"] + risk["risk_as_B"]) / 2

# ── STEP 5: Reduce to 2D with PCA ───────────────────────────────────────────
print("Running PCA...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# ── STEP 6: Build plot dataframe ─────────────────────────────────────────────
plot_df = pd.DataFrame({
    "airport":  classes,
    "x":        embeddings_2d[:, 0],
    "y":        embeddings_2d[:, 1],
})
plot_df = plot_df.merge(risk[["avg_risk"]], left_on="airport", right_index=True, how="left")
plot_df["avg_risk"] = plot_df["avg_risk"].fillna(0)

# ── STEP 7: Plot ─────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(
    plot_df["x"],
    plot_df["y"],
    c=plot_df["avg_risk"],
    cmap="RdYlGn_r",       # red = high risk, green = low risk
    s=100,
    alpha=0.8,
    edgecolors="black",
    linewidths=0.5
)

# Add airport labels for the top 20 riskiest airports
top_risk = plot_df.nlargest(20, "avg_risk")
for _, row in top_risk.iterrows():
    ax.annotate(
        row["airport"],
        (row["x"], row["y"]),
        fontsize=8,
        fontweight="bold",
        xytext=(5, 5),
        textcoords="offset points"
    )

# Add labels for a few low-risk airports too
low_risk = plot_df.nsmallest(10, "avg_risk")
for _, row in low_risk.iterrows():
    ax.annotate(
        row["airport"],
        (row["x"], row["y"]),
        fontsize=7,
        color="darkgreen",
        xytext=(5, 5),
        textcoords="offset points"
    )

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Average Cascade Risk Rate", fontsize=11)

ax.set_title("Airport Risk Embeddings — DFW Sequences (Jan 2025)\nColor = Cascade Risk (Red = High, Green = Low)", fontsize=13)
ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=10)
ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("airport_embeddings.png", dpi=150)
plt.show()
print("\nPlot saved to airport_embeddings.png")

# ── STEP 8: Print top 10 riskiest airport pairs ──────────────────────────────
print("\nTop 10 Riskiest Airport Pairs (A → DFW → B):")
pair_risk = df.groupby(["airport_A", "airport_B"])["CASCADING_DELAY"].agg(
    ["mean", "count"]
).reset_index()
pair_risk.columns = ["airport_A", "airport_B", "cascade_rate", "num_sequences"]
pair_risk = pair_risk[pair_risk["num_sequences"] >= 10]  # filter low sample pairs
pair_risk = pair_risk.sort_values("cascade_rate", ascending=False)
print(pair_risk.head(10).to_string(index=False))