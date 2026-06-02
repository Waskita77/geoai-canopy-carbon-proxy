from pathlib import Path

# File ini diasumsikan berada di folder scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

GLAD_FILE = DATA_ROOT / "GLAD" / "Forest_height_2019_SASIA.tif"
RAW_TRAIN_DIR = DATA_ROOT / "raw_train"
RAW_INFERENCE_DIR = DATA_ROOT / "raw_inference"
AOI_DIR = DATA_ROOT / "aoi"

PROCESSED_DIR = DATA_ROOT / "processed"
ALIGNED_LABEL_DIR = PROCESSED_DIR / "aligned_labels"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
TILES_DIR = DATA_ROOT / "tiles_dataset"

OUTPUT_DIR = DATA_ROOT / "output_maps"
HEIGHT_DIR = OUTPUT_DIR / "height"
CARBON_DIR = OUTPUT_DIR / "carbon"
REPORT_DIR = DATA_ROOT / "reports"

STATS_PATH = PROCESSED_DIR / "normalization_stats.npz"
MODEL_PATH = CHECKPOINT_DIR / "unet_canopy_model.pth"
TRAIN_LOG_PATH = REPORT_DIR / "training_log.csv"
SCENE_INDEX_PATH = PROCESSED_DIR / "tile_scene_index.csv"

EXPECTED_BANDS = 8
TILE_SIZE = 256
STRIDE = 256
NODATA = -9999.0
SEED = 42

# Training defaults
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 2e-4
VAL_RATIO = 0.15

# Carbon proxy parameters.
# Ini sengaja sederhana/indikatif. Ganti jika punya persamaan lokal/field plot.
AGB_COEF_A = 0.05
AGB_EXP_B = 2.2
CARBON_FRACTION = 0.47
CO2E_FACTOR = 44.0 / 12.0


def ensure_directories() -> None:
    for folder in [
        DATA_ROOT,
        DATA_ROOT / "GLAD",
        RAW_TRAIN_DIR,
        RAW_INFERENCE_DIR,
        AOI_DIR,
        PROCESSED_DIR,
        ALIGNED_LABEL_DIR,
        CHECKPOINT_DIR,
        TILES_DIR,
        HEIGHT_DIR,
        CARBON_DIR,
        REPORT_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
