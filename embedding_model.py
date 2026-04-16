import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── FILE PATH ────────────────────────────────────────────────────────────────
DATA_FILE = r"C:\Spring 2026\American Airlines Challenge\scripts\DFW_sequences_Jan2025.csv"
# ────────────────────────────────────────────────────────────────────────────

# ── STEP 1: Load data ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Total sequences: {len(df)}")
print(f"Cascade rate: {df['CASCADING_DELAY'].mean():.2%}")

# ── STEP 2: Encode airports as integer IDs ───────────────────────────────────
# The embedding layer needs integer IDs, not string codes like "ORD" or "MIA"
le = LabelEncoder()
all_airports = pd.concat([df["airport_A"], df["airport_B"]]).unique()
le.fit(all_airports)

df["airport_A_id"] = le.transform(df["airport_A"])
df["airport_B_id"] = le.transform(df["airport_B"])

num_airports = len(le.classes_)
print(f"Unique airports: {num_airports}")

# ── STEP 3: Prepare additional features ─────────────────────────────────────
# These are the non-embedding features passed alongside the airport embeddings
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

feature_cols = [
    "connection_minutes",
    "month_sin",
    "month_cos",
    "A_weather_delay_rate",
    "A_nas_delay_rate",
    "B_weather_delay_rate",
    "B_nas_delay_rate"
]

# Fill any missing values with 0
df[feature_cols] = df[feature_cols].fillna(0)

# ── STEP 4: Train/test split ─────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                      stratify=df["CASCADING_DELAY"])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# ── STEP 5: PyTorch Dataset ──────────────────────────────────────────────────
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ── STEP 6: Define the Embedding Model ──────────────────────────────────────
class AirportEmbeddingModel(nn.Module):
    def __init__(self, num_airports, embedding_dim=32, num_extra_features=7):
        super(AirportEmbeddingModel, self).__init__()

        # Each airport gets a 32-dimensional embedding vector
        self.embedding = nn.Embedding(num_airports, embedding_dim)

        # Dense layers after concatenating: airport_A_emb + airport_B_emb + extra features
        input_dim = embedding_dim * 2 + num_extra_features

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, airport_a, airport_b, features):
        emb_a = self.embedding(airport_a)   # shape: (batch, 32)
        emb_b = self.embedding(airport_b)   # shape: (batch, 32)
        x = torch.cat([emb_a, emb_b, features], dim=1)  # shape: (batch, 71)
        return self.network(x).squeeze(1)

model = AirportEmbeddingModel(num_airports=num_airports)
print(f"\nModel architecture:")
print(model)

# ── STEP 7: Handle class imbalance ──────────────────────────────────────────
# Weight the positive class higher since cascading delays are rare
n_neg = (df["CASCADING_DELAY"] == 0).sum()
n_pos = (df["CASCADING_DELAY"] == 1).sum()
pos_weight = torch.tensor([3.0], dtype=torch.float32)
print(f"\nClass balance — No cascade: {n_neg}, Cascade: {n_pos}")
print(f"Positive class weight: {pos_weight.item():.2f}")

criterion = nn.BCELoss(weight=None)  # We handle imbalance via pos_weight in training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── STEP 8: Training loop ────────────────────────────────────────────────────
EPOCHS = 50  # was 20, now 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  

print("\nTraining...")
train_losses = []  # ADD THIS LINE
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for airport_a, airport_b, features, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(airport_a, airport_b, features)

        # Apply pos_weight manually to handle class imbalance
        weights = torch.where(labels == 1, pos_weight.expand_as(labels), torch.ones_like(labels))
        loss = nn.functional.binary_cross_entropy(predictions, labels, weight=weights)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# ── STEP 9: Evaluation ───────────────────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for airport_a, airport_b, features, labels in test_loader:
        preds = model(airport_a, airport_b, features)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Convert probabilities to binary predictions at 0.5 threshold
binary_preds = (all_preds >= 0.3).astype(int)

print("\nClassification Report:")
print(classification_report(all_labels, binary_preds,
                             target_names=["No Cascade", "Cascade"]))
print(f"ROC-AUC Score:         {roc_auc_score(all_labels, all_preds):.4f}")
print(f"Precision-Recall AUC:  {average_precision_score(all_labels, all_preds):.4f}")

# ── STEP 10: Plot training loss ──────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()
print("\nTraining loss plot saved to training_loss.png")

# ── STEP 11: Save model and encoder ─────────────────────────────────────────
torch.save(model.state_dict(), r"C:\Spring 2026\American Airlines Challenge\scripts\embedding_model.pth")
np.save(r"C:\Spring 2026\American Airlines Challenge\scripts\airport_label_encoder.npy", le.classes_)
print("Model saved to embedding_model.pth")
print("Label encoder saved to airport_label_encoder.npy")