import os, sys, torch, logging
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_CPU  = DEVICE.type == 'cpu'
IS_CUDA = DEVICE.type == 'cuda'
print(f'Device: {DEVICE} | CUDA: {torch.cuda.is_available()}')

#BASE_DIR        = Path('/kaggle/working/ecg_project')
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ECG_IMAGE_DIR   = BASE_DIR / 'ecg_images'
RECORDS_DIR     = BASE_DIR / 'records500'
MODELS_DIR      = BASE_DIR / 'models'
LOGS_DIR        = BASE_DIR / 'logs'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
UPLOAD_FOLDER   = BASE_DIR / 'uploads'
DB_CSV          = BASE_DIR / 'ptbxl_database.csv'
SCP_CSV         = BASE_DIR / 'scp_statements.csv'
LOG_FILE        = LOGS_DIR / 'training.log'

for _d in [MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, UPLOAD_FOLDER]:
    _d.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
for _h in logging.getLogger().handlers:
    if isinstance(_h, logging.StreamHandler):
        try:
            _h.stream.reconfigure(errors='replace')
        except Exception:
            pass

SAMPLING_RATE       = 100
INPUT_SIGNAL_LEN    = 1000
OUTPUT_WAVEFORM_LEN = 500
NUM_LEADS           = 12

LATENT_DIM             = 512
NUM_CLASSES            = 5
NHEAD                  = 8
NUM_TRANSFORMER_LAYERS = 3
IMAGE_SIZE             = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

if IS_CUDA:
    BATCH_SIZE     = 16
    VAL_BATCH_SIZE = 32
    NUM_WORKERS    = 4
    PIN_MEMORY     = True
    NUM_EPOCHS     = 10
    WARMUP_EPOCHS  = 3
else:
    BATCH_SIZE     = 2
    VAL_BATCH_SIZE = 4
    NUM_WORKERS    = 0
    PIN_MEMORY     = False
    NUM_EPOCHS     = 6
    WARMUP_EPOCHS  = 2

FINETUNE_EPOCHS = NUM_EPOCHS - WARMUP_EPOCHS
TOTAL_EPOCHS    = NUM_EPOCHS

LEARNING_RATE      = 1e-4
FINETUNE_LR        = 1e-6
LOSS_ALPHA         = 0.7
LOSS_BETA          = 0.3
WEIGHT_DECAY       = 1e-4
GRADIENT_CLIP      = 1.0
ACCUMULATION_STEPS = 2 if IS_CPU else 1
MAX_RECORDS        = 500 if IS_CPU else None

VALIDATION_SPLIT        = 0.15
TEST_SPLIT              = 0.15
EARLY_STOPPING_PATIENCE = 4
SAVE_BEST_ONLY          = True

MEMORY_CLEANUP_FREQ        = 5
USE_AMP                    = IS_CUDA
USE_GRADIENT_CHECKPOINTING = IS_CPU

LOG_INTERVAL      = 5
VERBOSE           = True
USE_PROGRESS_BARS = True

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'Normal ECG',
    'MI'  : 'Myocardial Infarction',
    'STTC': 'ST/T-Wave Change',
    'CD'  : 'Conduction Disturbance',
    'HYP' : 'Hypertrophy',
}
CLASS_MAP          = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SCP_SUPERCLASS_COL = 'diagnostic_class'

def verify_paths():
    logger.info('='*70)
    logger.info('VERIFYING PATHS')
    logger.info('='*70)
    checks = {
        'ECG Images'        : ECG_IMAGE_DIR,
        'ptbxl_database.csv': DB_CSV,
        'scp_statements.csv': SCP_CSV,
    }
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        logger.info(f"[{'OK' if exists else 'MISSING'}] {name}: {path}")
        if not exists:
            all_ok = False
    logger.info('='*70)
    return all_ok

def print_config():
    logger.info('='*70)
    logger.info('CONFIGURATION SUMMARY')
    logger.info('='*70)
    logger.info(f'Device: {DEVICE} | CPU Mode: {IS_CPU}')
    logger.info(f'Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS}')
    logger.info(f'Epochs: {NUM_EPOCHS} (Warmup: {WARMUP_EPOCHS}, Finetune: {FINETUNE_EPOCHS})')
    logger.info('Split: 70% train / 15% val / 15% test')
    logger.info(f'Early Stopping Patience: {EARLY_STOPPING_PATIENCE}')
    logger.info(f'Max Records: {MAX_RECORDS}')
    logger.info('='*70)