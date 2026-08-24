# optimized_tiler_vips.py
import os
import json
import numpy as np
import pandas as pd
import pyvips
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import cv2
from src.preprocess.masking.tissue_cells import (
    get_all_tissue_bboxes,
    get_tiling_grid,
    filter_cells_by_mask_fast
)
from src.preprocess.profiling import profile
import config

_worker_slide = None
_worker_path = None


# -------------------------------
# Initializer function to open a file only once
# -------------------------------
def _init_worker(img_path):
    global _worker_slide, _worker_path
    _worker_slide = pyvips.Image.new_from_file(img_path, access='random')
    _worker_path = img_path

# -------------------------------
# Function that prepares tile to respect jpeg characteristics
# -------------------------------
def _prepare_tile_for_jpeg(tile: pyvips.Image) -> pyvips.Image:
    if tile.bands > 3:
        tile = tile.extract_band(0, n=3)

    if tile.format != "uchar":
        tile = tile.cast("uchar")

    return tile

# -------------------------------
# Worker function (pickle-safe)
# -------------------------------
@profile
def process_tile_batch(args):
    """
    Worker to save a batch of tiles using pyvips.
    args = (img_path, output_dir, tile_size, batch_tiles, tissue_thr)
    """
    img_path, output_dir, tile_size, batch_tiles, tissue_thr, slide_id, resolution_level = args
    saved = 0
    saved_ids = []

    # Open slide once
    #slide = pyvips.Image.new_from_file(img_path, access='random')

    for (x, y, tile_id) in batch_tiles:
        # crop region
        try:
            region = _worker_slide.crop(x, y, tile_size, tile_size)
        except Exception:
            continue

        # Convert to numpy for tissue check
        tile_np = np.ndarray(buffer=region.write_to_memory(),
                             dtype=np.uint8,
                             shape=[region.height, region.width, region.bands])

        if tile_np.shape[2] > 3:
            tile_np = tile_np[:, :, :3]

        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        tissue_ratio = np.mean(gray < 220)
        if tissue_ratio < tissue_thr:
            continue

        # Save jpg
        tile_name = f"{slide_id}_res{resolution_level}_tile_{tile_id:06d}_x{x}_y{y}.jpg"
        path = os.path.join(output_dir, tile_name)
        cv2.imwrite(path, cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]) # use the img already loaded in memory
        #region = _prepare_tile_for_jpeg(region)
        #region.write_to_file(path, Q=90)
        saved_ids.append((tile_id, float(tissue_ratio)))   # <-- was: saved_ids.append(tile_id)
        saved += 1

    return saved_ids #saved


# -------------------------------
# MemorySafeTiler using pyvips
# -------------------------------
@profile
class MemorySafeTiler:
    def __init__(self, img_path, mask_lowres, tile_size=224, stride=224,
                 tissue_threshold=0.1, output_dir="./tiles", mask_level=None,
                 batch_size=1024, slide_id=None, resolution_level=0):
        self.img_path = img_path
        self.tile_size = tile_size
        self.stride = stride
        self.tissue_threshold = tissue_threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.batch_size = batch_size
        self.slide_id = slide_id
        self.resolution_level = resolution_level

        self.mask_lowres_01 = (mask_lowres > 0).astype(np.uint8)

        # Open slide to get downsample and dimensions
        slide = pyvips.Image.new_from_file(img_path, access='sequential')
        self.W_full, self.H_full = slide.width, slide.height
        if mask_level is None:
            self.mask_level = min(3, int(np.log2(max(self.W_full, self.H_full) / mask_lowres.shape[1])))
        else:
            self.mask_level = mask_level
        self.ds = self.W_full / mask_lowres.shape[1]

        self.tile_low = max(1, int(round(self.tile_size / self.ds)))
        self.cell_size_low = max(self.tile_low * 4, 32)
    @profile
    def tile(self, save_tiles=True, num_workers=None):
        if num_workers is None:
            num_workers = int(getattr(config, "NUM_WORKERS", 4))

        mask_for_cc = (self.mask_lowres_01 * 255).astype(np.uint8)
        bboxes_lowres = get_all_tissue_bboxes(mask_for_cc,
                                              min_area_ratio=float(getattr(config, "MIN_AREA_RATIO", 0.0)))
        if not bboxes_lowres:
            return {"image": self.img_path, "total_tiles": 0, "saved_tiles": 0, "skipped_tiles": 0,
                    "save_ratio": 0.0, "metadata_path": None, "note": "No tissue bboxes detected."}

        # Grid of cells in low-res
        cells_lowres = get_tiling_grid(bboxes_lowres, self.cell_size_low)
        valid_cells_lowres = filter_cells_by_mask_fast(cells_lowres, mask_for_cc,
                                                       tissue_threshold=self.tissue_threshold,
                                                       tile_size=self.tile_low)
        if not valid_cells_lowres:
            return {"image": self.img_path, "total_tiles": 0, "saved_tiles": 0, "skipped_tiles": 0,
                    "save_ratio": 0.0, "metadata_path": None, "note": "No valid cells after filtering."}

        # Map to full-res
        valid_cells_fullres = [
            (int(round(x * self.ds)), int(round(y * self.ds)),
             int(round(w * self.ds)), int(round(h * self.ds)))
            for (x, y, w, h) in valid_cells_lowres
        ]

        # Prepare tasks
        records = []
        batches = []
        batch_tiles = []
        tile_id = 0

        for bx, by, bw, bh in valid_cells_fullres:
            x_end = bx + bw - self.tile_size
            y_end = by + bh - self.tile_size
            for y in range(by, y_end + 1, self.stride):
                for x in range(bx, x_end + 1, self.stride):
                    tile_filename = f"{self.slide_id}_res{self.resolution_level}_tile_{tile_id:06d}_x{x}_y{y}.jpg"
                    records.append({
                    "tile_id": tile_id,
                    "slide_id": self.slide_id, 
                    "resolution_level":self.resolution_level,
                    "tile_filename": tile_filename,
                    "tile_file_path": os.path.join(self.output_dir, tile_filename),
                    "left": x, "top": y,
                    "right": x + self.tile_size, "bottom": y + self.tile_size,
                    "kept": False #default change if saved
                    })
                    batch_tiles.append((x, y, tile_id))
                    tile_id += 1
                    if len(batch_tiles) >= self.batch_size:
                        batches.append(batch_tiles)
                        batch_tiles = []
        if batch_tiles:
            batches.append(batch_tiles)

        saved = 0
        if save_tiles:
            args_list = [(self.img_path, self.output_dir, self.tile_size, batch, self.tissue_threshold, self.slide_id, self.resolution_level)
                         for batch in batches]
            saved_ids_all = []
            with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(self.img_path, )) as executor:
                for result in executor.map(process_tile_batch, args_list):
                    #saved += result
                    saved_ids_all.extend(result)
            saved = len(saved_ids_all)
            tissue_ratio_by_id = {tid: tr for (tid, tr) in saved_ids_all}   # <-- new
            saved_set = set(tissue_ratio_by_id.keys())                     # <-- was: set(saved_ids_all)
            #saved_set = set(saved_ids_all)
            for r in records:
                r["kept"] = r["tile_id"] in saved_set

        metadata_path = os.path.join(self.output_dir, "tiles_metadata.csv")
        pd.DataFrame(records).to_csv(metadata_path, index=False)
        
	# Save JSON metadata (same structure as crop_tiles)
        json_metadata = {
            "wsi_name": os.path.basename(self.img_path),
            "tile_size": self.tile_size,
            "tiles": [
                {
                    "tile_id": r["tile_id"],
                    "filename": r["tile_filename"],
                    "x": r["left"],
                    "y": r["top"],
                    "width": self.tile_size,
                    "height": self.tile_size,
                    "tissue_ratio": tissue_ratio_by_id.get(r["tile_id"])
                }
                for r in records if r["kept"]
            ]
        }
        json_path = os.path.join(self.output_dir, f"{self.slide_id}_tiles.json")

        with open(json_path, 'w') as f:
            json.dump(json_metadata, f, indent=4)
        print(f"Saved {saved} tiles and metadata to {json_path}")


        return {"image": self.img_path, "mask_level": self.mask_level, "downsample": self.ds,
                "total_tiles": tile_id, "saved_tiles": saved, "skipped_tiles": tile_id - saved,
                "save_ratio": (saved / tile_id if tile_id else 0.0), "metadata_path": metadata_path,
                "note": "OK"}
