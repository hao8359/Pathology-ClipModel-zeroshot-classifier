import os
import glob
import time
import multiprocessing
import numpy as np
import cv2
from openslide import OpenSlide
import tempfile
import shutil
import csv
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from src.preprocess.masking.week2 import double_pass_and_p1wr, otsu_wo_blur
from src.preprocess.masking.tissue_cells import get_all_tissue_bboxes
from src.preprocess.tiling.optimized_tiler_vips import MemorySafeTiler
from src.preprocess.profiling import profile
from src.preprocess.blur.blur_fast2 import run_blur_fast
import config
from src.preprocess.anonimization import get_anon_slide_id, load_mapping
from src.preprocess.temp_file import anonymized_batch, anonymized_slide

def _as_int(x, default: int) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _as_float(x, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

@profile
def find_svs_files(folder: str):
    """Recursively find all .svs files in a folder."""
    return sorted(glob.glob(os.path.join(folder, "**", "*.svs"), recursive=True))

@profile
def make_lowres_mask(thumbnail_rgb: np.ndarray) -> np.ndarray:
    """Return a low-res binary tissue mask {0,1}."""
    method = (config.MASK_METHOD or "").lower()

    if method == "double_pass_and_p1wr":
        mask = double_pass_and_p1wr(thumbnail_rgb)
        mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask.astype(np.uint8)
    elif method == "otsu_wo_blur":
        mask = otsu_wo_blur(thumbnail_rgb)
        mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask.astype(np.uint8)
    else:
        raise NotImplementedError(f"Mask method '{config.MASK_METHOD}' not implemented")

def run_per_slide_anon():
    svs_folder = config.SINGLE_SVS_FOLDER
    if not svs_folder or not os.path.isdir(svs_folder):
        print("SINGLE_SVS_FOLDER is not a valid folder:", svs_folder)
        return

    svs_files = find_svs_files(svs_folder)
    if not svs_files:
        print("No .svs files found in folder:", svs_folder)
    
    base_output_root = os.path.join(config.OUTPUT_DIR, "pipeline")
    scratch_output_root = config.OUTPUT_DIR_PIPELINE # scratch, e.g. $TMPDIR/output
    master_csv_path = os.path.join(base_output_root, "blur_results.csv")  # the one file


    case_mapping = load_mapping()
    work_dir = tempfile.mkdtemp(prefix="anon_")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = []
            for slide_path in svs_files:
                with anonymized_slide(slide_path, work_dir, case_mapping) as local_svs:
                    slide_output_dir = tiling_step(local_svs)      # tile -> returns output_dir
                    run_blur_fast(input_pipeline_dir=slide_output_dir, csv_output_dir=slide_output_dir) #per-slide on scratch

                    # kick off copy-back on a background thread, keep looping immediately
                    pending.append(
                        executor.submit(stage_out, slide_output_dir, base_output_root, scratch_output_root, master_csv_path)
                    )
                # local_svs is deleted here (temp .svs freed), tiles for this slide
                # continue copying back in the background while next slide starts

            for fut in as_completed(pending):
                fut.result()   # raise on any failed copy
    except KeyboardInterrupt:
        print("[run_per_slide_anon] Interrupted shutting down.")
        raise
        
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    

def tiling_step(slide_path):
    # Parse config once
    num_workers = _as_int(getattr(config, "NUM_WORKERS", multiprocessing.cpu_count()), multiprocessing.cpu_count())
    tile_width = _as_int(config.TILE_WIDTH, 224)
    tile_stride = _as_int(config.TILE_STRIDE_X, 224)
    batch_size = _as_int(config.BATCH_SIZE, 1024)
    tissue_thr = _as_float(config.TISSUE_THRESHOLD, 0.2)

    out_root = config.OUTPUT_DIR_PIPELINE
    if not out_root:
        print("OUTPUT_DIR_PIPELINE is not set in .env/config")
        return

    slide_start = time.time()
        
    # Output directory per slide and name anonimization
    slide_name = os.path.basename(slide_path)
    anon_slide_id = slide_name[:36]
    output_dir = os.path.join(out_root, slide_name, "tiles")
    os.makedirs(output_dir, exist_ok=True)
        
    print(f"\nProcessing slide: {anon_slide_id}")

    # Open slide and create low-res thumbnail
    with OpenSlide(slide_path) as slide:
        mask_level = min(3, slide.level_count - 1)
        W, H = slide.level_dimensions[mask_level]
        thumbnail_rgb = np.array(slide.read_region((0, 0), mask_level, (W, H)))[:, :, :3]

    # Generate tissue mask (low-res) -> {0,1}
    mask_lowres = make_lowres_mask(thumbnail_rgb)


    # Initialize tiler
    tiler = MemorySafeTiler(
        img_path=slide_path,
        mask_lowres=mask_lowres,
        tile_size=tile_width,
        stride=tile_stride,
        tissue_threshold=tissue_thr,
        output_dir=output_dir,
        mask_level=mask_level,
        batch_size=batch_size,
        slide_id=anon_slide_id,	   
        resolution_level=0   
    )

    # Tile and save
    summary = tiler.tile(save_tiles=True, num_workers=num_workers)

    # Print essential info only
    print(f"Finished: {anon_slide_id}")
    print(f"Tiles saved: {summary['saved_tiles']} / {summary['total_tiles']}")
    print(f"Metadata CSV: {summary['metadata_path']}")
    print(f" Slide time: {time.time() - slide_start:.2f} seconds")
    
    return output_dir



_csv_lock = threading.Lock()


def merge_slide_csv(slide_csv_path, master_csv_path, lock):
    """
    Append one slide's blur_results.csv into the single master CSV.
    Thread-safe: guarded by `lock` since multiple slides may finish
    staging around the same time.
    """
    if not os.path.isfile(slide_csv_path):
        print(f"[stage_out] WARNING: no blur CSV found at {slide_csv_path}, skipping merge")
        return

    with open(slide_csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if not reader:
        return

    header, rows = reader[0], reader[1:]

    with lock:
        master_exists = os.path.isfile(master_csv_path)
        os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)
        with open(master_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not master_exists:
                writer.writerow(header)
            writer.writerows(rows)


def stage_out(local_dir, base_output_root, scratch_output_root, master_csv_path):
    rel = os.path.relpath(local_dir, scratch_output_root)
    dst = os.path.join(base_output_root, rel)
    print(f"Copying {local_dir}")
    print(f"      -> {dst}")

    # Copy tiles, but don't duplicate the per-slide CSV into permanent
    # storage, it gets merged into the single master CSV instead.
    shutil.copytree(
        local_dir, dst, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("blur_results.csv"),
    )

    slide_csv_path = os.path.join(local_dir, "blur_results.csv")
    merge_slide_csv(slide_csv_path, master_csv_path, _csv_lock)

    return rel

def stage_out_no_blur(local_dir, base_output_root, scratch_output_root):
    rel = os.path.relpath(local_dir, scratch_output_root)
    dst = os.path.join(base_output_root, rel)
    print(f"Copying {local_dir}")
    print(f"      -> {dst}")
    shutil.copytree(local_dir, dst, dirs_exist_ok=True)
    return rel
    
def run_per_slide_anon_only_tiling():
    svs_folder = config.SINGLE_SVS_FOLDER
    if not svs_folder or not os.path.isdir(svs_folder):
        print("SINGLE_SVS_FOLDER is not a valid folder:", svs_folder)
        return
    svs_files = find_svs_files(svs_folder)
    if not svs_files:
        print("No .svs files found in folder:", svs_folder)
    
    base_output_root = os.path.join(config.OUTPUT_DIR, "pipeline")
    scratch_output_root = config.OUTPUT_DIR_PIPELINE # scratch, e.g. $TMPDIR/output
    # master_csv_path = os.path.join(base_output_root, "blur_results.csv")  # not needed, no blur
    # case_mapping = load_mapping()
    # work_dir = tempfile.mkdtemp(prefix="anon_")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = []
            for slide_path in svs_files:
                # with anonymized_slide(slide_path, work_dir, case_mapping) as local_svs:
                slide_output_dir = tiling_step(slide_path)      # tile -> returns output_dir
                # run_blur_fast(input_pipeline_dir=slide_output_dir, csv_output_dir=slide_output_dir) #per-slide on scratch
                # kick off copy-back on a background thread, keep looping immediately
                pending.append(
                    executor.submit(stage_out_no_blur, slide_output_dir, base_output_root, scratch_output_root)
                )
                # local_svs is deleted here (temp .svs freed), tiles for this slide
                # continue copying back in the background while next slide starts
            for fut in as_completed(pending):
                fut.result()   # raise on any failed copy
    except KeyboardInterrupt:
        print("[run_per_slide_anon] Interrupted shutting down.")
        raise
        
    finally:
        # shutil.rmtree(work_dir, ignore_errors=True)  # no work_dir anymore
        pass

