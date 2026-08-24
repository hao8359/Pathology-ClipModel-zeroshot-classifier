# run_pipeline_vips_tiles_only.py
import time
print(f"PYTHON_ENTRY {time.time()}", flush=True)

import os
import time
import pandas as pd
import config

print(f"BEFORE IMPORTS {time.time()}", flush=True)
from src.preprocess.tiling.run_all_slides_fast import run_per_slide_anon, run_per_slide_anon_only_tiling
#print(f"AFTER IMPORTING TILING {time.time()}", flush=True)
from src.preprocess.blur.blur_fast2 import run_blur_fast
#print(f"AFTER IMPORTING BLUR {time.time()}", flush=True)
from src.preprocess.profiling import profile
#print(f"AFTER IMPORTING PROFILING {time.time()}", flush=True)


print(f"IMPORTS_DONE {time.time()}", flush=True)

print(f"PIPELINE_START {time.time()}", flush=True)

@profile
def create_dataset_summary():
    """Collect per-slide metadata CSVs and merge them into one dataset CSV."""
    pipeline_root = config.OUTPUT_DIR_PIPELINE
    if not pipeline_root:
        return

    metadata_files = []
    for root, _, files in os.walk(pipeline_root):
        for f in files:
            if f == "tiles_metadata.csv":
                metadata_files.append(os.path.join(root, f))

    if not metadata_files:
        print("No metadata CSV files found for dataset summary.")
        return

    dfs = [pd.read_csv(f) for f in metadata_files]
    dataset = pd.concat(dfs, ignore_index=True)
    
    out_eval = config.OUTPUT_DIR_TILES_EV or os.path.join(pipeline_root, "_sum")
    os.makedirs(out_eval, exist_ok=True)

    out_csv = os.path.join(out_eval, "dataset_summary.csv")
    dataset.to_csv(out_csv, index=False)

    print(f"Dataset summary saved to: {out_csv}")


if __name__ == "__main__":
    t0 = time.time()
    
    run_per_slide_anon()

    #run_per_slide_anon_only_tiling()

    #print("\nRunning blur detection...")
    #run_blur_fast()

    #create_dataset_summary()

    print(f"\nTOTAL PIPELINE TIME: {(time.time() - t0):.2f} seconds")
    
print(f"PIPELINE_DONE {time.time()}", flush=True)
