# src/preprocess/config.py

import os
from dotenv import load_dotenv

# Load .env from current working directory
load_dotenv()

BASE_PATH = os.getenv("BASE_PATH")
OUTPUT_DIR = os.path.join(BASE_PATH, "output")
#DATA_DIR = os.path.join(BASE_PATH, "data1", "svs")
DATA_DIR = os.path.join(BASE_PATH, "test_data", "svs")

OUTPUT_DIR_TILES = os.path.join(OUTPUT_DIR, "tiles")
OUTPUT_DIR_TILES_EV = os.path.join(OUTPUT_DIR, "tiles_ev")
OUTPUT_DIR_PIPELINE = os.path.join(OUTPUT_DIR, "pipeline")
OUTPUT_DIR_DATA = os.path.join(OUTPUT_DIR, "slide_metadata")
SCRATCH_DIR = (
    os.getenv("SCRATCH_DIR") or
    os.getenv("TMPDIR") or
    os.path.join(BASE_PATH, "scratch")
)
OUTPUT_DIR_PIPELINE = os.path.join(SCRATCH_DIR, "output", "pipeline")  # scratch ? tiling writes here
ANON_MAPPING_FILE = os.path.join(OUTPUT_DIR_DATA, "case_mapping.json")

SIF_PATH = os.getenv("SIF_PATH")

SINGLE_SVS_FOLDER = DATA_DIR

# Access variables from .env

PATCH_WIDTH=int(os.getenv("PATCH_WIDTH", 224))
PATCH_HEIGHT=int(os.getenv("PATCH_HEIGHT", 224))


TILE_WIDTH=int(os.getenv("TILE_WIDTH", 224))
TILE_HEIGHT=int(os.getenv("TILE_HEIGHT", 224))

TILE_STRIDE_X=int(os.getenv("TILE_STRIDE_X", TILE_WIDTH))
TILE_STRIDE_Y=int(os.getenv("TILE_STRIDE_Y", TILE_HEIGHT))

TISSUE_THRESHOLD=float(os.getenv("TISSUE_THRESHOLD", 0.2))

IMAGE_FOLDER=os.getenv("IMAGE_FOLDER")
MASK_FOLDER=os.getenv("MASK_FOLDER")

LEVEL=int(os.getenv("LEVEL", 0))

X_COORD=int(os.getenv("X_COORD", 0))
Y_COORD=int(os.getenv("Y_COORD", 0))

MASK_METHOD=os.getenv("MASK_METHOD")

THRESH_TENENGRAD = float(os.getenv("THRESH_TENENGRAD", 4500.0))
THRESH_LAPLACIAN=float(os.getenv("THRESH_LAPLACIAN", 110.0))
THRESH_TISSUE_BLUR=float(os.getenv("THRESH_TISSUE_BLUR", 0.3))

MIN_AREA_RATIO=float(os.getenv("MIN_AREA_RATIO", 0.00001))
NUM_WORKERS=int(os.getenv("NUM_WORKERS", 32))
BATCH_SIZE=int(os.getenv("BATCH_SIZE", 512))

