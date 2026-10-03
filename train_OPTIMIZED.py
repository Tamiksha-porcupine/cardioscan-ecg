import os, gc, ast, time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging

from config_OPTIMIZED import (
    DEVICE, IS_CPU, IS_CUDA,
    BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    NUM_EPOCHS, WARMUP_EPOCHS, TOTAL_EPOCHS,
    LEARNING_RATE, FINETUNE_LR, WEIGHT_DECAY, GRADIENT_CLIP,
    LOSS_ALPHA, LOSS_BETA, ACCUMULATION_STEPS,
    MAX_RECORDS, EARLY_STOPPING_PATIENCE, SAVE_BEST_ONLY,
    MEMORY_CLEANUP_FREQ, USE_PROGRESS_BARS,
    IMAGE_SIZE, NORM_MEAN, NORM_STD,
    OUTPUT_WAVEFORM_LEN, INPUT_SIGNAL_LEN,
    ECG_IMAGE_DIR, RECORDS_DIR, DB_CSV, SCP_CSV, MODELS_DIR,
    BASE_DIR, CLASS_NAMES, SCP_SUPERCLASS_COL,
    logger, print_config, verify_paths,
    VALIDATION_SPLIT, TEST_SPLIT
)
from model_OPTIMIZED import UltimateECGHybrid, build_model

SUPERCLASS_MAP = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    def __init__(self, records, meta, scp_df, transform=None):
        self.records   = records
        self.meta      = meta
        self.scp_df    = scp_df
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        # Pre-cache labels
        self.labels = [self._get_label(rid) for rid in self.records]

    def _get_label(self, ecg_id):
        try:
            codes = self.meta.loc[ecg_id].scp_codes
            for code, confidence in codes.items():
                if code in self.scp_df.index:
                    sc = self.scp_df.loc[code][SCP_SUPERCLASS_COL]
                    if isinstance(sc, str) and sc in SUPERCLASS_MAP:
                        return SUPERCLASS_MAP[sc]
        except Exception:
            pass
        return 0  # default NORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rid = self.records[idx]
        # Load image
        img_path = ECG_IMAGE_DIR / f"{rid}.png"
        try:
            with Image.open(img_path) as img:
                img_t = self.transform(img.convert("RGB"))
        except Exception:
            img_t = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        # Load waveform target
        try:
            raw = str(self.meta.loc[rid].filename_hr).replace("\\", "/")
            if "records500/" in raw:
                fname = raw.split("records500/", 1)[-1]
            else:
                fname = raw.split("/")[-1]
            import wfdb
            rpath  = RECORDS_DIR / fname
            signal = wfdb.rdrecord(str(rpath)).p_signal[:, 0]
            total  = INPUT_SIGNAL_LEN + OUTPUT_WAVEFORM_LEN
            if len(signal) < total:
                signal = np.pad(signal, (0, total - len(signal)))
            target = signal[INPUT_SIGNAL_LEN:total]
        except Exception:
            target = np.zeros(OUTPUT_WAVEFORM_LEN)

        return (
            img_t,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_backbone_trainable(model, trainable):
    for p in model.encoder.vit.parameters():
        p.requires_grad = trainable
    for p in model.encoder.eff.parameters():
        p.requires_grad = trainable
    status = "UNFROZEN" if trainable else "FROZEN"
    n = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"[OK] Backbones {status} | Trainable params: {n:,}")


def run_epoch(model, loader, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss, count = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, tgts, lbls) in enumerate(
            tqdm(loader, desc="Train" if training else "Val ",
                 disable=not USE_PROGRESS_BARS, leave=False)
        ):
            imgs = imgs.to(DEVICE)
            tgts = tgts.to(DEVICE)
            lbls = lbls.to(DEVICE)

            w_pred, logits = model(imgs)
            loss = (LOSS_ALPHA * F.mse_loss(w_pred, tgts) +
                    LOSS_BETA  * F.cross_entropy(logits, lbls))

            if training:
                (loss / ACCUMULATION_STEPS).backward()
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if (batch_idx + 1) % MEMORY_CLEANUP_FREQ == 0:
                    gc.collect()

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def save_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses,   label="Val Loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info(f"[OK] Loss curve saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    logger.info("="*80)
    logger.info("ECG HYBRID AI - TWO-PHASE TRAINING")
    logger.info("="*80)
    print_config()
    verify_paths()

    # Load metadata
    logger.info("Loading dataset metadata...")
    meta = pd.read_csv(DB_CSV, index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)
    scp_df = pd.read_csv(SCP_CSV, index_col=0)

    all_records = meta[meta.filename_hr.notnull()].index.tolist()
    if MAX_RECORDS:
        all_records = all_records[:MAX_RECORDS]
    logger.info(f"[OK] Total records: {len(all_records)}")

    # Train / Val / Test split
    import random
    random.seed(42)
    random.shuffle(all_records)
    n       = len(all_records)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VALIDATION_SPLIT)
    n_train = n - n_val - n_test

    train_recs = all_records[:n_train]
    val_recs   = all_records[n_train:n_train + n_val]
    test_recs  = all_records[n_train + n_val:]

    logger.info(f"Split -> Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    # Datasets
    logger.info("Building datasets (caching labels)...")
    train_ds = ECGDataset(train_recs, meta, scp_df)
    val_ds   = ECGDataset(val_recs,   meta, scp_df)
    test_ds  = ECGDataset(test_recs,  meta, scp_df)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=VAL_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Model
    model = build_model(eval_mode=False)

    # Phase 1 optimizer (frozen backbones)
    set_backbone_trainable(model, False)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_loss     = float("inf")
    patience_counter  = 0
    train_losses      = []
    val_losses        = []

    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)

    for epoch in range(NUM_EPOCHS):
        epoch_num = epoch + 1

        # Phase transition
        if epoch == WARMUP_EPOCHS:
            logger.info("-"*60)
            logger.info("PHASE 2: Unfreezing backbones for fine-tuning")
            logger.info("-"*60)
            set_backbone_trainable(model, True)
            optimizer = optim.AdamW(
                model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
            )

        phase = "Frozen" if epoch < WARMUP_EPOCHS else "Finetune"
        logger.info(f"Epoch {epoch_num}/{NUM_EPOCHS} [{phase}]")

        train_loss = run_epoch(model, train_loader, optimizer, training=True)
        val_loss   = run_epoch(model, val_loader,   training=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Gap: {abs(train_loss - val_loss):.4f}"
        )

        # Overfit/underfit hint
        gap = val_loss - train_loss
        if gap > 0.1:
            logger.info(f"  [NOTE] Gap={gap:.4f} -- possible overfitting")
        elif train_loss > 0.5 and val_loss > 0.5:
            logger.info(f"  [NOTE] Both losses high -- possible underfitting")
        else:
            logger.info(f"  [NOTE] Training looks healthy")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model_path = MODELS_DIR / "ecg_best.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"  [OK] Best model saved (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch_num}")
                break

        gc.collect()

    # Save loss curves
    curve_path = MODELS_DIR / "loss_curves.png"
    save_loss_curves(train_losses, val_losses, curve_path)

    # ── Final test evaluation ──────────────────────────────────────────────
    logger.info("="*80)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*80)

    # Load best model
    best_path = MODELS_DIR / "ecg_best.pth"
    if best_path.exists():
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state)
        logger.info("[OK] Best model loaded for test evaluation")

    test_loss = run_epoch(model, test_loader, training=False)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Overfitting analysis
    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    logger.info("="*80)
    logger.info("OVERFITTING / UNDERFITTING ANALYSIS")
    logger.info("="*80)
    logger.info(f"Final Train Loss : {final_train:.4f}")
    logger.info(f"Final Val Loss   : {final_val:.4f}")
    logger.info(f"Test Loss        : {test_loss:.4f}")
    logger.info(f"Train-Val Gap    : {abs(final_train - final_val):.4f}")

    if final_val > final_train + 0.1:
        logger.info("[RESULT] OVERFITTING detected -- val loss much higher than train loss")
        logger.info("  Suggestions: add dropout, reduce model size, add data augmentation")
    elif final_train > 0.5 and final_val > 0.5:
        logger.info("[RESULT] UNDERFITTING detected -- both losses are high")
        logger.info("  Suggestions: train longer, increase model capacity, lower LR")
    else:
        logger.info("[RESULT] Model is well-fitted -- train and val losses are close")

    # Save results
    results = {
        "train_losses"  : train_losses,
        "val_losses"    : val_losses,
        "test_loss"     : test_loss,
        "best_val_loss" : best_val_loss,
        "epochs_run"    : len(train_losses),
    }
    results_path = BASE_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[OK] Results saved to {results_path}")
    logger.info("[SUCCESS] TRAINING FINISHED SUCCESSFULLY!")

    return True


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)