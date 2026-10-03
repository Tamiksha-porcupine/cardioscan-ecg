# ═══════════════════════════════════════════════════════════════════════════════
# predict.py - ECG Prediction Inference Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   python predict.py --image path/to/ecg_image.png
#   python predict.py --record path/to/ecg_record   (wfdb .hea/.dat pair)
#   python predict.py --batch  path/to/image_folder/
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (mirrors config_OPTIMIZED.py — no import needed for standalone use)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI"  : "Myocardial Infarction",
    "STTC": "ST/T-Wave Change",
    "CD"  : "Conduction Disturbance",
    "HYP" : "Hypertrophy",
}

DEFAULT_WEIGHTS = Path(__file__).parent / "models" / "ecg_best.pth"

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str = None):
    """Load the trained UltimateECGHybrid model."""
    from model_OPTIMIZED import build_model

    path = weights_path or str(DEFAULT_WEIGHTS)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Weights not found at: {path}\n"
            f"Run training first or pass --weights <path>"
        )

    print(f"Loading model from: {path}")
    model = build_model(weights_path=path, eval_mode=True)
    model.to(DEVICE)
    model.eval()
    print(f"Model ready on {DEVICE}\n")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# INPUT LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_from_image(image_path: str) -> torch.Tensor:
    """Load a pre-rendered ECG PNG and return a (1,3,224,224) tensor."""
    img = Image.open(image_path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0).to(DEVICE)


def load_from_wfdb(record_path: str) -> torch.Tensor:
    """
    Load a raw WFDB record (.hea + .dat), render lead-I to image,
    and return a (1,3,224,224) tensor.
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb:  pip install wfdb")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0]  # Lead I

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(signal[:1000], color="black", linewidth=0.7)
    ax.axis("off")
    ax.set_facecolor("white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    return TRANSFORM(img).unsqueeze(0).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, tensor: torch.Tensor) -> dict:
    """
    Run inference on a single (1,3,224,224) tensor.

    Returns a dict with:
        predicted_class     : str   e.g. "NORM"
        description         : str   e.g. "Normal ECG"
        confidence          : float  0-100
        probabilities       : dict  {class: prob%}
        predicted_waveform  : list  (500 float values)
    """
    with torch.no_grad():
        waveform, logits = model(tensor)

    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class"   : CLASS_NAMES[pred_idx],
        "description"       : CLASS_DESCRIPTIONS[CLASS_NAMES[pred_idx]],
        "confidence"        : round(float(probs[pred_idx]) * 100, 2),
        "probabilities"     : {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASS_NAMES, probs)
        },
        "predicted_waveform": waveform.squeeze().cpu().numpy().tolist(),
    }


def predict_batch(model, image_folder: str) -> list:
    """Run inference on all PNGs in a folder. Returns list of result dicts."""
    folder = Path(image_folder)
    images = sorted(folder.glob("*.png"))
    if not images:
        print(f"No PNG files found in {folder}")
        return []

    results = []
    for img_path in images:
        tensor = load_from_image(str(img_path))
        result = predict_single(model, tensor)
        result["file"] = img_path.name
        results.append(result)
        print(f"  {img_path.name:40s} → {result['predicted_class']}  ({result['confidence']:.1f}%)")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# REPORT SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_report(result: dict, output_dir: str = "."):
    """Save prediction result as JSON + waveform plot PNG."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    # JSON report
    json_path = out / "prediction_report.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Report saved  → {json_path}")

    # Waveform plot
    waveform = np.array(result["predicted_waveform"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Waveform
    axes[0].plot(waveform, color="#1f77b4", linewidth=1.0)
    axes[0].set_title("Predicted Next Waveform (500 samples)", fontsize=12)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Probability bar chart
    classes = list(result["probabilities"].keys())
    probs   = list(result["probabilities"].values())
    colors  = ["#2ecc71" if c == result["predicted_class"] else "#95a5a6" for c in classes]
    bars = axes[1].bar(classes, probs, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Class Probabilities (%)", fontsize=12)
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_ylim(0, 105)
    for bar, prob in zip(bars, probs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{prob:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    fig.suptitle(
        f"ECG Prediction: {result['predicted_class']} — {result['description']}  "
        f"(confidence {result['confidence']:.1f}%)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    plot_path = out / "prediction_plot.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved    → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ECG Prediction — UltimateECGHybrid inference"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a pre-rendered ECG PNG")
    group.add_argument("--record", type=str, help="Path to WFDB record (no extension)")
    group.add_argument("--batch",  type=str, help="Folder of ECG PNGs for batch inference")

    p.add_argument("--weights", type=str, default=None,
                   help="Path to ecg_best.pth (default: models/ecg_best.pth)")
    p.add_argument("--output",  type=str, default="predictions",
                   help="Output folder for reports/plots (default: predictions/)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip saving report and plot")
    return p.parse_args()


def main():
    args = parse_args()
    model = load_model(args.weights)

    if args.batch:
        print(f"\nBatch mode: {args.batch}")
        results = predict_batch(model, args.batch)
        if results and not args.no_report:
            out = Path(args.output)
            out.mkdir(exist_ok=True, parents=True)
            batch_json = out / "batch_results.json"
            with open(batch_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nBatch report saved → {batch_json}")

        # Summary
        from collections import Counter
        counts = Counter(r["predicted_class"] for r in results)
        print("\n── Batch Summary ──────────────────────────")
        for cls, count in counts.most_common():
            print(f"  {cls:6s} ({CLASS_DESCRIPTIONS[cls]:30s}): {count}")
        print(f"  Total: {len(results)} records")

    else:
        # Single prediction
        if args.image:
            print(f"Input image : {args.image}")
            tensor = load_from_image(args.image)
        else:
            print(f"Input record: {args.record}")
            tensor = load_from_wfdb(args.record)

        result = predict_single(model, tensor)

        # Print result
        print("\n" + "═" * 50)
        print("  ECG PREDICTION RESULT")
        print("═" * 50)
        print(f"  Diagnosis   : {result['predicted_class']} — {result['description']}")
        print(f"  Confidence  : {result['confidence']:.1f}%")
        print()
        print("  Class Probabilities:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob / 5)
            marker = " ◄" if cls == result["predicted_class"] else ""
            print(f"    {cls:6s} {prob:5.1f}%  {bar}{marker}")
        print("═" * 50)

        if not args.no_report:
            save_report(result, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
