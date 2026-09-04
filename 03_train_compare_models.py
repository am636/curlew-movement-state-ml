from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "curlew_ml_table.csv"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
TABLES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "longitude",
    "latitude",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "previous_speed_kmh",
    "bio1",
    "bio12",
    "elevation_m",
]

np.random.seed(42)
torch.manual_seed(42)


class SmallMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 24),
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_torch_model(x_train, y_train, x_test):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True)
    model = SmallMLP(x_train.shape[1])

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor(negatives / positives) if positives > 0 else torch.tensor(1.0)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x_test)).numpy()
    return probability, model


def metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, probability) if len(np.unique(y_true)) > 1 else np.nan
    average_precision = average_precision_score(y_true, probability) if np.any(y_true == 1) else np.nan
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
    }


data = pd.read_csv(DATA)
X = data[FEATURES].to_numpy(dtype=float)
y = data["active_movement"].to_numpy(dtype=int)
groups = data["bird_id"].astype(str).to_numpy()

n_birds = len(np.unique(groups))
if n_birds < 2:
    raise ValueError("At least two birds are needed for grouped validation.")

cv = GroupKFold(n_splits=n_birds)
predictions = {
    "logistic": np.zeros(len(data)),
    "random_forest": np.zeros(len(data)),
    "pytorch_mlp": np.zeros(len(data)),
}
forest_importances = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    held_out = np.unique(groups[test_idx])
    print(f"Fold {fold}: held out {', '.join(held_out)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X[train_idx])
    x_test_scaled = scaler.transform(X[test_idx])

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    logistic.fit(x_train_scaled, y[train_idx])
    predictions["logistic"][test_idx] = logistic.predict_proba(x_test_scaled)[:, 1]

    forest = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X[train_idx], y[train_idx])
    predictions["random_forest"][test_idx] = forest.predict_proba(X[test_idx])[:, 1]
    held_out_importance = permutation_importance(
        forest,
        X[test_idx],
        y[test_idx],
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    forest_importances.append(held_out_importance.importances_mean)

    torch_probability, _ = fit_torch_model(
        x_train_scaled,
        y[train_idx],
        x_test_scaled,
    )
    predictions["pytorch_mlp"][test_idx] = torch_probability

importance_table = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": np.mean(forest_importances, axis=0),
    "importance_sd": np.std(forest_importances, axis=0, ddof=1),
}).sort_values("importance_mean", ascending=False)
importance_table.round(6).to_csv(
    TABLES / "random_forest_permutation_importance.csv",
    index=False,
)

pred_table = data[["row_id", "bird_id", "active_movement"]].copy()
for model_name, probability in predictions.items():
    pred_table[f"{model_name}_probability"] = probability
pred_table.to_csv(TABLES / "oof_predictions.csv", index=False)

per_bird = []
for bird, subset in pred_table.groupby("bird_id"):
    truth = subset["active_movement"].to_numpy()
    for model_name in predictions:
        probability = subset[f"{model_name}_probability"].to_numpy()
        row = {"bird_id": bird, "model": model_name}
        row.update(metrics(truth, probability))
        per_bird.append(row)
per_bird_table = pd.DataFrame(per_bird)
per_bird_table.round(6).to_csv(TABLES / "per_bird_metrics.csv", index=False)

metric_columns = [
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "brier_score",
]
metrics_table = (
    per_bird_table.groupby("model", sort=False)[metric_columns]
    .mean()
    .reset_index()
)
metrics_table.round(6).to_csv(TABLES / "model_metrics.csv", index=False)

pooled_rows = []
for model_name, probability in predictions.items():
    row = {"model": model_name}
    row.update(metrics(y, probability))
    pooled_rows.append(row)
pd.DataFrame(pooled_rows).round(6).to_csv(TABLES / "pooled_metrics.csv", index=False)

# Fit one final PyTorch model to all rows after cross-validation.
# This saved model is for reproducibility/use, not for reporting model performance.
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
_, final_torch_model = fit_torch_model(X_scaled, y, X_scaled)
torch.save(
    {
        "state_dict": final_torch_model.state_dict(),
        "features": FEATURES,
        "scaler_mean": torch.tensor(final_scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(final_scaler.scale_, dtype=torch.float32),
    },
    MODELS / "pytorch_mlp_state.pt",
)

print("\nModel comparison")
print(metrics_table.round(3).to_string(index=False))
print("Saved final PyTorch state to:", MODELS / "pytorch_mlp_state.pt")
