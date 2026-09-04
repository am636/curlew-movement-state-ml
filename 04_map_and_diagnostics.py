from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "curlew_ml_table.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

tracks = pd.read_csv(DATA)
pred = pd.read_csv(TABLES / "oof_predictions.csv")
metrics = pd.read_csv(TABLES / "model_metrics.csv")
per_bird = pd.read_csv(TABLES / "per_bird_metrics.csv")
importance = pd.read_csv(TABLES / "random_forest_permutation_importance.csv")

model_labels = {
    "logistic": "Logistic regression",
    "random_forest": "Random Forest",
    "pytorch_mlp": "PyTorch MLP",
}
models = list(model_labels)

plt.figure(figsize=(8, 6))
points = plt.scatter(
    tracks["longitude"],
    tracks["latitude"],
    c=tracks["active_movement"],
    s=3,
    alpha=0.5,
)
cbar = plt.colorbar(points, ticks=[0, 1])
cbar.ax.set_yticklabels(["low movement", "active"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Curlew GPS locations and operational movement state")
plt.tight_layout()
plt.savefig(FIGURES / "movement_state_map.png", dpi=180)
plt.close()

plot_data = metrics.set_index("model")[["roc_auc", "average_precision", "balanced_accuracy"]]
plot_errors = per_bird.groupby("model")[["roc_auc", "average_precision", "balanced_accuracy"]].std()
plot_data = plot_data.rename(index=model_labels, columns={
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "balanced_accuracy": "Balanced accuracy",
})
plot_errors = plot_errors.rename(index=model_labels, columns={
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "balanced_accuracy": "Balanced accuracy",
}).reindex(plot_data.index)
plot_data.plot(kind="bar", yerr=plot_errors, capsize=3, figsize=(8, 5))
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("")
plt.title("Leave-one-bird-out performance (mean ± SD)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "model_comparison.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 6))
for model in models:
    observed, predicted = calibration_curve(
        pred["active_movement"],
        pred[f"{model}_probability"],
        n_bins=10,
        strategy="quantile",
    )
    plt.plot(predicted, observed, marker="o", label=model_labels[model])
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("Predicted probability")
plt.ylabel("Observed active fraction")
plt.title("Calibration diagnostic")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "calibration.png", dpi=180)
plt.close()

for model in models:
    probability = pred[f"{model}_probability"].to_numpy()
    predicted = (probability >= 0.5).astype(int)
    cm = confusion_matrix(pred["active_movement"], predicted)
    disp = ConfusionMatrixDisplay(cm, display_labels=["low movement", "active"])
    disp.plot(values_format="d")
    plt.title(model_labels[model])
    plt.tight_layout()
    plt.savefig(FIGURES / f"confusion_{model}.png", dpi=180)
    plt.close()

bird_f1 = per_bird.pivot(index="bird_id", columns="model", values="f1")
bird_f1 = bird_f1.rename(columns=model_labels)
bird_f1.plot(kind="bar", figsize=(9, 5))
plt.ylim(0, 1)
plt.ylabel("F1 score")
plt.xlabel("Bird")
plt.title("Performance for each held-out bird")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "per_bird_f1.png", dpi=180)
plt.close()

importance = importance.sort_values("importance_mean")
ax = importance.plot(
    x="feature",
    y="importance_mean",
    xerr="importance_sd",
    kind="barh",
    legend=False,
    capsize=3,
    figsize=(7, 5),
)
ax.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean decrease in held-out balanced accuracy")
plt.ylabel("")
plt.title("Permutation importance (mean ± SD across birds)")
plt.tight_layout()
plt.savefig(FIGURES / "random_forest_permutation_importance.png", dpi=180)
plt.close()

print("Saved figures to:", FIGURES)
