# ═══════════════════════════════════════════════════════════════════════════════
# pre_render_ecg_OPTIMIZED.py - Comprehensive Data Preparation (Optimized)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Optimizations:
# ✅ Efficient image rendering (matplotlib)
# ✅ Error recovery
# ✅ Progress tracking
# ✅ Memory management
# ✅ Comprehensive logging
# ✅ Batch processing
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import pandas as pd

# Fix Windows console (cp1252) crashing on ✓/⚠️/✅ characters in log output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (faster, no GUI)
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path
from tqdm import tqdm
import ast
import logging
import gc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(os.getcwd())
DB_CSV = BASE_DIR / "ptbxl_database.csv"
RECORDS_DIR = BASE_DIR / "records500"
OUTPUT_DIR = BASE_DIR / "ecg_images"
LOG_FILE = BASE_DIR / "pre_render.log"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Backup safety net: if the console still can't encode a character for any
# reason, replace it instead of crashing the whole pre-rendering run.
for _h in logging.getLogger().handlers:
    if isinstance(_h, logging.StreamHandler) and _h.stream in (sys.stdout, sys.stderr):
        try:
            _h.stream.reconfigure(errors="replace")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE RENDERING (Optimized)
# ─────────────────────────────────────────────────────────────────────────────

def signal_to_image(signal, output_path, figsize=(3, 2), dpi=100):
    """
    Convert raw ECG signal to standardized image.
    
    Optimizations:
    - Close figures to free memory
    - Use non-interactive backend
    - Minimal processing
    - Error handling
    """
    try:
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Plot signal
        ax.plot(signal[:1000], color="black", linewidth=0.7)
        ax.set_facecolor("white")
        ax.axis('off')
        
        # Save
        fig.savefig(
            output_path,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi
        )
        
        # Close and cleanup
        plt.close(fig)
        
        return True
    
    except Exception as e:
        logger.warning(f"Error rendering image for {output_path}: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PRE-RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main pre-rendering pipeline"""
    
    logger.info("\n" + "="*70)
    logger.info("ECG DATA PRE-RENDERING (OPTIMIZED)")
    logger.info("="*70)
    
    # Step 1: Verify database exists
    if not DB_CSV.exists():
        logger.error(f"Database file not found: {DB_CSV}")
        return False
    
    logger.info(f"Reading database: {DB_CSV}")
    
    try:
        meta = pd.read_csv(DB_CSV, index_col="ecg_id")
        logger.info(f"[OK] Loaded {len(meta)} records from database")
    except Exception as e:
        logger.error(f"Error reading database: {e}")
        return False
    
    # Step 2: Initialize statistics
    stats = {
        "total": len(meta),
        "existing": 0,
        "rendered": 0,
        "failed": 0,
        "missing_signal": 0
    }
    
    # Step 3: Process records with progress bar
    logger.info("\nProcessing records...")
    
    for ecg_id, row in tqdm(meta.iterrows(), total=len(meta), 
                             desc="Pre-rendering ECG images", 
                             disable=False):
        
        try:
            output_path = OUTPUT_DIR / f"{ecg_id}.png"
            
            # Check if image already exists
            if output_path.exists():
                stats["existing"] += 1
                continue
            
            # Get signal file path
            try:
                # filename_hr in ptbxl_database.csv looks like:
                #   "records500/00000/00001_hr"
                # RECORDS_DIR already points at the local "records500" folder,
                # so we must keep the subfolder (e.g. "00000/00001_hr") and only
                # strip the leading "records500/" prefix - NOT the whole path.
                raw_fname = str(row.filename_hr).replace("\\", "/")
                if "records500/" in raw_fname:
                    fname = raw_fname.split("records500/", 1)[-1]
                else:
                    fname = raw_fname
                record_path = RECORDS_DIR / fname
                
                # Check if signal files exist
                if not (Path(str(record_path) + ".hea").exists() and 
                       Path(str(record_path) + ".dat").exists()):
                    stats["missing_signal"] += 1
                    continue
                
                # Load signal
                record = wfdb.rdrecord(str(record_path))
                signal = record.p_signal[:, 0]  # Use first lead
                
                # Render to image
                if signal_to_image(signal, str(output_path)):
                    stats["rendered"] += 1
                else:
                    stats["failed"] += 1
            
            except Exception as e:
                logger.debug(f"Error processing record {ecg_id}: {e}")
                stats["failed"] += 1
        
        except Exception as e:
            logger.error(f"Unexpected error for {ecg_id}: {e}")
            stats["failed"] += 1
        
        # Memory cleanup every 100 records
        if (stats["rendered"] + stats["existing"]) % 100 == 0:
            gc.collect()
    
    # Step 4: Print summary
    logger.info("\n" + "="*70)
    logger.info("PRE-RENDERING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total records in database:     {stats['total']}")
    logger.info(f"Existing images (skipped):     {stats['existing']}")
    logger.info(f"New images rendered:           {stats['rendered']}")
    logger.info(f"Missing signal files:          {stats['missing_signal']}")
    logger.info(f"Failed/errors:                 {stats['failed']}")
    logger.info(f"Total images in folder:        {len(list(OUTPUT_DIR.glob('*.png')))}")
    logger.info("="*70)
    
    # Verification
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION")
    logger.info("="*70)
    
    total_images = len(list(OUTPUT_DIR.glob("*.png")))
    expected = len(meta)
    coverage = (total_images / expected) * 100
    
    logger.info(f"Image coverage: {coverage:.1f}% ({total_images}/{expected})")
    
    if coverage < 90:
        logger.warning(f"[WARNING] Coverage below 90%! Check for missing records.")
    else:
        logger.info(f"[OK] Coverage sufficient for training")
    
    logger.info("="*70)
    logger.info("\nDataset is ready for training!")
    
    return True

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset():
    """Verify dataset integrity"""
    logger.info("\nVerifying dataset...")
    
    checks = {
        "ECG Images folder": OUTPUT_DIR,
        "Records folder": RECORDS_DIR,
        "Database CSV": DB_CSV,
    }
    
    all_ok = True
    for name, path in checks.items():
        exists = Path(path).exists()
        status = "[OK]" if exists else "[MISSING]"
        logger.info(f"{status} {name}: {path}")
        if not exists:
            all_ok = False
    
    if all_ok:
        logger.info("[OK] All required files present")
    else:
        logger.warning("[MISSING] Some required files missing")
    
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("ECG PRE-RENDERING PIPELINE (OPTIMIZED)")
    logger.info("="*70)
    
    # Verify dataset first
    if not verify_dataset():
        logger.error("Dataset verification failed!")
        sys.exit(1)
    
    # Run pre-rendering
    success = main()
    
    if success:
        logger.info("\n[SUCCESS] Pre-rendering completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n[FAILED] Pre-rendering failed!")
        sys.exit(1)