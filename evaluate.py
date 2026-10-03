"""
evaluate.py - Model Performance Evaluation
==========================================
Runs your trained model on the test set and prints:
  - Classification: F1, Precision, Recall, Accuracy, ROC-AUC (per class + overall)
  - Waveform prediction: RMSE, MAE, R² score
  - Confusion matrix (saved as PNG)
  - Full report saved to evaluation_report.json

Usage:
    python evaluate.py
"""

import os, sys, ast, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

# ── Import your project files ─────────────────────────────────────────────────
from config_OPTIMIZED import (
    DEVICE, BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    MAX_RECORDS, DB_CSV, SCP_CSV, ECG_IMAGE_DIR, RECORDS_DIR,
    MODELS_DIR, BASE_DIR, CLASS_NAMES, SCP_SUPERCLASS_COL,
    OUTPUT_WAVEFORM_LEN, INPUT_SIGNAL_LEN,
    VALIDATION_SPLIT, TEST_SPLIT, logger
)
from model_OPTIMIZED import build_model
from train_OPTIMIZED import ECGDataset

SUPERCLASS_MAP = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
WEIGHTS_PATH   = MODELS_DIR / "ecg_best.pth"
OUT_DIR        = BASE_DIR / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CardioScan — Model Evaluation")
print("="*65)

model = build_model(weights_path=str(WEIGHTS_PATH), eval_mode=True)
model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 2. REBUILD TEST SET (same split as training)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRebuilding test split...")
import random
meta   = pd.read_csv(DB_CSV, index_col="ecg_id")
meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)
scp_df = pd.read_csv(SCP_CSV, index_col=0)

all_records = meta[meta.filename_hr.notnull()].index.tolist()
if MAX_RECORDS:
    all_records = all_records[:MAX_RECORDS]

random.seed(42)
random.shuffle(all_records)
n       = len(all_records)
n_test  = int(n * TEST_SPLIT)
n_val   = int(n * VALIDATION_SPLIT)
n_train = n - n_val - n_test

test_recs = all_records[n_train + n_val:]
print(f"Test set size: {len(test_recs)} records")

test_ds     = ECGDataset(test_recs, meta, scp_df)
test_loader = DataLoader(test_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ─────────────────────────────────────────────────────────────────────────────
# 3. RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning inference on test set...")

all_labels, all_preds, all_probs = [], [], []
all_wave_true, all_wave_pred     = [], []

with torch.no_grad():
    for imgs, targets, labels in tqdm(test_loader, desc="Evaluating"):
        imgs    = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        labels  = labels.to(DEVICE)

        wave_pred, logits = model(imgs)
        probs = F.softmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_wave_true.extend(targets.cpu().numpy())
        all_wave_pred.extend(wave_pred.cpu().numpy())

        gc.collect()

all_labels    = np.array(all_labels)
all_preds     = np.array(all_preds)
all_probs     = np.array(all_probs)
all_wave_true = np.array(all_wave_true)
all_wave_pred = np.array(all_wave_pred)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CLASSIFICATION METRICS")
print("="*65)

accuracy  = accuracy_score(all_labels, all_preds)
f1_macro  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
f1_weight = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
f1_per    = f1_score(all_labels, all_preds, average=None,       zero_division=0)
prec_per  = precision_score(all_labels, all_preds, average=None,zero_division=0)
rec_per   = recall_score(all_labels, all_preds, average=None,   zero_division=0)

# ROC-AUC (one-vs-rest)
try:
    roc_auc = roc_auc_score(
        np.eye(len(CLASS_NAMES))[all_labels], all_probs,
        average="macro", multi_class="ovr"
    )
except Exception:
    roc_auc = None

print(f"\n  Overall Accuracy      : {accuracy*100:.2f}%")
print(f"  F1 Score (macro)      : {f1_macro:.4f}")
print(f"  F1 Score (weighted)   : {f1_weight:.4f}")
if roc_auc:
    print(f"  ROC-AUC (macro OvR)   : {roc_auc:.4f}")

print(f"\n  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'-'*50}")
for i, cls in enumerate(CLASS_NAMES):
    support = int((all_labels == i).sum())
    print(f"  {cls:<8} {prec_per[i]:>10.4f} {rec_per[i]:>10.4f} {f1_per[i]:>10.4f} {support:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. WAVEFORM METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  WAVEFORM PREDICTION METRICS")
print("="*65)

rmse = np.sqrt(np.mean((all_wave_true - all_wave_pred) ** 2))
mae  = np.mean(np.abs(all_wave_true - all_wave_pred))

# R² score
ss_res = np.sum((all_wave_true - all_wave_pred) ** 2)
ss_tot = np.sum((all_wave_true - all_wave_true.mean()) ** 2)
r2     = 1 - (ss_res / (ss_tot + 1e-8))

# MAPE (mean absolute percentage error)
nonzero = all_wave_true != 0
mape = np.mean(np.abs((all_wave_true[nonzero] - all_wave_pred[nonzero])
                       / all_wave_true[nonzero])) * 100 if nonzero.any() else None

print(f"\n  RMSE (Root Mean Square Error) : {rmse:.6f}")
print(f"  MAE  (Mean Absolute Error)    : {mae:.6f}")
print(f"  R²   (Coefficient of Det.)    : {r2:.4f}")
if mape:
    print(f"  MAPE (Mean Abs % Error)       : {mape:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6. WHAT DO THESE NUMBERS MEAN?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  WHAT THESE NUMBERS MEAN")
print("="*65)

print(f"""
  ACCURACY ({accuracy*100:.1f}%)
  → Out of every 100 ECGs, the model correctly diagnosed {accuracy*100:.0f}.
    Anything above 70% on a 5-class medical dataset is solid.

  F1 SCORE (macro: {f1_macro:.3f} | weighted: {f1_weight:.3f})
  → F1 balances precision and recall. Score ranges from 0 to 1.
    > 0.75 = good  |  > 0.85 = very good  |  > 0.90 = excellent
    Macro treats all classes equally (important for rare conditions).
    Weighted accounts for class imbalance.

  ROC-AUC ({f'{roc_auc:.3f}' if roc_auc else 'N/A'})
  → Measures how well the model separates classes. 
    0.5 = random guessing  |  1.0 = perfect separation
    > 0.80 is considered good for medical AI.

  RMSE ({rmse:.6f}) & MAE ({mae:.6f})
  → How far off the predicted waveform is from the real one.
    Lower is better. These are in the same units as your ECG signal amplitude.

  R² ({r2:.4f})
  → How much of the waveform variation the model explains.
    1.0 = perfect  |  0.0 = no better than predicting the mean
    > 0.7 is generally considered good.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("Saving confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Confusion Matrix (counts)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("Actual", fontsize=11)

# Normalised
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix (normalised)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("Actual", fontsize=11)

plt.suptitle("Model Performance — Confusion Matrix", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {cm_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PER-CLASS F1 BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
colors = ["#2D9B6F","#D94F3D","#E07B2A","#5B7FD4","#9B59B6"]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_NAMES, f1_per, color=colors, width=0.5,
              edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, f1_per):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(f1_macro, color="#333", linestyle="--", linewidth=1.2, label=f"Macro avg: {f1_macro:.3f}")
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("F1 Score per Class", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2, linestyle="--")
plt.tight_layout()
f1_path = OUT_DIR / "f1_per_class.png"
fig.savefig(f1_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {f1_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE JSON REPORT
# ─────────────────────────────────────────────────────────────────────────────
report = {
    "overall": {
        "accuracy"   : round(accuracy, 4),
        "f1_macro"   : round(f1_macro, 4),
        "f1_weighted": round(f1_weight, 4),
        "roc_auc"    : round(roc_auc, 4) if roc_auc else None,
    },
    "per_class": {
        cls: {
            "precision": round(float(prec_per[i]), 4),
            "recall"   : round(float(rec_per[i]),  4),
            "f1"       : round(float(f1_per[i]),   4),
            "support"  : int((all_labels == i).sum()),
        }
        for i, cls in enumerate(CLASS_NAMES)
    },
    "waveform": {
        "rmse": round(float(rmse), 6),
        "mae" : round(float(mae),  6),
        "r2"  : round(float(r2),   4),
        "mape": round(float(mape), 2) if mape else None,
    },
    "files": {
        "confusion_matrix": str(cm_path),
        "f1_per_class"    : str(f1_path),
    }
}

report_path = OUT_DIR / "evaluation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved → {report_path}")

print("\n" + "="*65)
print("  EVALUATION COMPLETE")
print(f"  All files saved to: {OUT_DIR}")
print("="*65 + "\n")
