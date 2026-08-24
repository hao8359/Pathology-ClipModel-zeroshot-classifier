# src/preprocess/blur/blur_fast2.py
import csv
import shutil
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.preprocess.profiling import profile
from src.preprocess.anonimization import get_anon_slide_id
import cv2
import numpy as np
import signal
import sys
import config
import time
import threading

BG_THRESHOLD = 235
MIN_BLOCK_TISSUE = 0.10
BLOCK_SIZE = 56

# --- timeout / termination behavior -----------------------------------
TILE_TIMEOUT_SEC = 300     # if a single tile takes longer than this, treat it as hung
POOL_POLL_INTERVAL = 1.0   # how often (sec) to check on pending tasks
 
class _ProcessKiller:
    kill_now = False
    _signal_count = 0

    def __init__(self):
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        self._signal_count += 1
        print(f"\n[blur_fast] Caught signal {signum} (#{self._signal_count}) ? "
              f"stopping after current batch, killing worker pool")
        self.kill_now = True
        if self._signal_count >= 3:
            print("[blur_fast] Repeated signals and graceful shutdown hasn't "
                  "completed -- forcing immediate exit.")
            os._exit(1)
	
@profile
def get_tissue_mask(img: np.ndarray) -> np.ndarray:
    """Return a binary mask where tissue pixels are True."""
    return img < BG_THRESHOLD

@profile
def get_tenengrad(img: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Compute Tenengrad score using only tissue pixels."""
    if np.mean(tissue_mask) < 0.01:
        return 0.0

    img_float = img.astype(np.float32)
    sobelx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = sobelx**2 + sobely**2

    return float(np.mean(grad_sq[tissue_mask]))

@profile
def get_laplacian(img: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Compute Laplacian variance using only tissue pixels."""
    if np.mean(tissue_mask) < 0.01:
        return 0.0

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(np.var(laplacian[tissue_mask]))

@profile
def get_block_tenengrad(img: np.ndarray) -> float:
    """
    For low-tissue tiles:
    split the tile into blocks, keep only blocks with enough tissue,
    compute Tenengrad on each valid block, and return the 75th percentile.
    """
    h, w = img.shape
    scores = []

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            block = img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            if block.size == 0:
                continue

            block_mask = block < BG_THRESHOLD
            if np.mean(block_mask) < MIN_BLOCK_TISSUE:
                continue

            score = get_tenengrad(block, block_mask)
            scores.append(score)

    if not scores:
        return 0.0

    return float(np.percentile(scores, 75))


    
cv2.setNumThreads(0) # prevents OpenCV from spawning its own threads at all, removing the fork hazard
@profile
def process_single_tile(task_data):
    """
    task_data = (
        slide_id,
        tile_name,
        tile_path,
        thresh_tenengrad,
        thresh_laplacian,
        thresh_tissue_blur
    )
    """
    slide_id, tile_name, tile_path, thresh_t, thresh_l, thresh_tissue = task_data

    img = cv2.imread(tile_path, 0)
    if img is None:
        return None

    tissue_mask = get_tissue_mask(img)
    tissue_fraction = float(np.mean(tissue_mask))

    # Case 1: normal tissue content
    if tissue_fraction >= thresh_tissue:
        ten_score = get_tenengrad(img, tissue_mask)
        lap_score = get_laplacian(img, tissue_mask)
        method_used = "global_tissue"

        # Need BOTH metrics to say blurry
        is_blurry = (ten_score <= thresh_t) and (lap_score <= thresh_l)

    # Case 2: low tissue content
    else:
        ten_score = get_block_tenengrad(img)
        lap_score = get_laplacian(img, tissue_mask)
        method_used = "block_tissue"

        # Use block Tenengrad only for the final decision
        is_blurry = ten_score <= thresh_t

    is_sharp = not is_blurry
    status = "Sharp" if is_sharp else "Blurry"

    row = [
        slide_id,
        tile_name,
        f"{tissue_fraction:.4f}",
        f"{ten_score:.2f}",
        f"{lap_score:.2f}",
        method_used,
        status,
        tile_path,
    ]

    return row, is_sharp

def _worker_init():
    # Workers must NOT inherit the parent's custom SIGTERM handler via fork.
    # If they do, pool.terminate()'s SIGTERM (and any SIGTERM the scheduler
    # sends) gets caught and merely sets a flag nobody checks, instead of
    # actually killing the worker -- which is why workers were surviving
    # termination and flooding the log with repeated "Caught signal" prints.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _make_pool(num_workers):
    return mp.Pool(processes=num_workers, maxtasksperchild=50, initializer=_worker_init)

def _force_kill_pool(pool, join_timeout=15.0):
    """
    Terminate a pool, but never trust join() to actually return: a worker
    blocked on I/O (e.g. a stalled network filesystem) sits in an
    uninterruptible sleep (D-state) where no signal -- not even SIGKILL --
    is delivered until the underlying syscall returns. In that case
    pool.join() can hang forever. This bounds the wait and, if exceeded,
    SIGKILLs the worker PIDs directly and gives up on join() rather than
    hanging the whole process.
    """
    worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p.pid]

    try:
        pool.terminate()
    except Exception as e:
        print(f"[blur_fast] pool.terminate() raised: {e}")

    join_thread_done = False

    def _join():
        nonlocal join_thread_done
        try:
            pool.join()
        finally:
            join_thread_done = True

    t = threading.Thread(target=_join, daemon=True)
    t.start()
    t.join(timeout=join_timeout)

    if not join_thread_done:
        print(f"[blur_fast] WARNING: pool.join() did not return within "
              f"{join_timeout}s -- worker(s) likely stuck on blocking I/O "
              f"(check `ps -eo pid,stat,cmd` for STAT=D on this node). "
              f"Force-killing worker PIDs directly: {worker_pids}")
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[blur_fast] Could not SIGKILL pid {pid}: {e}")
		
def _run_tasks_with_timeout(tasks, num_workers, per_task_timeout, killer):
    """
    Runs `tasks` in a multiprocessing.Pool.
 
    Unlike ProcessPoolExecutor, this can actually recover from a hung task:
    if any single tile exceeds `per_task_timeout`, the whole pool is
    terminated (which kills the stuck worker process) and restarted, then
    processing continues with the remaining tasks. This avoids the deadlock
    where the pool's shutdown/exit blocks forever waiting on a task that
    will never finish.
 
    Returns (results, timed_out_tile_names, was_interrupted)
    """
    results = []
    timed_out_tiles = []
    pending = list(tasks)
 
    while pending and not killer.kill_now:
        pool = _make_pool(num_workers)
        async_results = {}
        submit_times = {}
 
        try:
            for i, t in enumerate(pending):
                if killer.kill_now:
                    print("[blur_fast] Interrupt received during submission ? "
                          "stopping early.")
                    break
                async_results[i] = pool.apply_async(process_single_tile, (t,))
                submit_times[i] = time.monotonic()
            pool.close()  # no more tasks will be added to this pool instance
 
            done_idx = set()
            hung_idx = None
 
            while len(done_idx) < len(async_results) and not killer.kill_now:
                progressed = False
                for i, ar in async_results.items():
                    if i in done_idx:
                        continue
                    if ar.ready():
                        try:
                            res = ar.get(timeout=0)
                            if res:
                                results.append(res)
                        except Exception as e:
                            print(f"[blur_fast] ERROR processing '{pending[i][1]}': {e}")
                        done_idx.add(i)
                        progressed = True
                    elif time.monotonic() - submit_times[i] > per_task_timeout:
                        hung_idx = i
                        break
 
                if hung_idx is not None or killer.kill_now:
                    break
                if not progressed:
                    time.sleep(POOL_POLL_INTERVAL)
 
            if killer.kill_now:
                print("[blur_fast] Interrupt received terminating pool now.")
                break
 
            if hung_idx is not None:
                tile_name = pending[hung_idx][1]
                print(f"[blur_fast] WARNING: tile '{tile_name}' exceeded "
                      f"{per_task_timeout}s terminating and restarting the pool.")
                timed_out_tiles.append(tile_name)
                # drop the hung tile, keep everything else that hadn't finished yet
                pending = [pending[i] for i in range(len(pending))
                           if i not in done_idx and i != hung_idx]
                continue
            else:
                pending = []
        finally:
            # Always force-terminate: guarantees no orphaned worker processes
            # survive, whether we exit normally, via timeout, or via a crash.
            #pool.terminate()
            #pool.join()
            _force_kill_pool(pool)
 
    return results, timed_out_tiles, killer.kill_now
    
@profile
def run_blur_fast(
    input_pipeline_dir: str | None = None,
    csv_output_dir: str | None = None,
    thresh_tenengrad: float | None = None,
    thresh_laplacian: float | None = None,
    thresh_tissue_blur: float | None = None,
) -> None:
    """
    Blur filtering for tiles saved under the pipeline output.

    - Scans: <pipeline_dir>/**/tiles/*.jpg
    - Moves blurry tiles to: <pipeline_dir>/<slide_id>/blurry_tiles/
    - Writes: <pipeline_dir>/blur_results.csv
    """
    pipeline_dir = input_pipeline_dir or config.OUTPUT_DIR_PIPELINE
    if not pipeline_dir:
        raise ValueError("OUTPUT_DIR_PIPELINE is not set in .env/config")
    # Default CSV location: pipeline folder
    if csv_output_dir is None:
        csv_output_dir = config.OUTPUT_DIR_PIPELINE

    thresh_t = float(
        thresh_tenengrad if thresh_tenengrad is not None else config.THRESH_TENENGRAD
    )
    thresh_l = float(
        thresh_laplacian if thresh_laplacian is not None else config.THRESH_LAPLACIAN
    )
    thresh_tissue = float(
        thresh_tissue_blur if thresh_tissue_blur is not None else config.THRESH_TISSUE_BLUR
    )

    pipeline_path = Path(pipeline_dir)
    #pipeline_path.mkdir(parents=True, exist_ok=True)

    #csv_path = pipeline_path / "blur_results.csv"
    csv_dir = Path(csv_output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "blur_results.csv"

    slide_id = pipeline_path.parent.name
    print(f"[blur_fast] Scanning tiles under: {pipeline_dir}")
    print(f"[blur_fast] Output CSV: {csv_path}")
    print(
        f"[blur_fast] THRESH_TENENGRAD={thresh_t} | "
        f"THRESH_LAPLACIAN={thresh_l} | "
        f"THRESH_TISSUE_BLUR={thresh_tissue}"
    )

    #tile_paths = list(pipeline_path.rglob("tiles/*.jpg"))
    tile_paths = list(pipeline_path.rglob("*.jpg"))
    if not tile_paths:
        #print("[blur_fast] No tiles found (tiles/*.jpg). Nothing to do.")
        print("[blur_fast] No tiles found (*.jpg). Nothing to do.")
        return

    tasks = [
        (p.parents[1].name, p.name, str(p), thresh_t, thresh_l, thresh_tissue)
        for p in tile_paths
    ]

    total_tasks = len(tasks)
    

    print(f"[blur_fast] Found {total_tasks} tiles.")
    
    killer = _ProcessKiller()
    num_workers = getattr(config, "NUM_WORKERS", mp.cpu_count())
    
    results, timed_out_tiles, was_interrupted = _run_tasks_with_timeout(
        tasks, num_workers, TILE_TIMEOUT_SEC, killer
    )
    
    #write csv
    sharp_count = 0
    blurry_count = 0
        
    csv_exists = Path(csv_path).exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not csv_exists:
            writer.writerow([
                "slide_id",
                "tile_name",
                "tissue_fraction",
                "tenengrad_score",
                "laplacian_score",
                "method_used",
                "status",
                "tile_path",
            ])

    #with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    #    writer = csv.writer(f)
    #    writer.writerow([
    #        "slide_id",
    #        "tile_name",
    #        "tissue_fraction",
    #        "tenengrad_score",
    #        "laplacian_score",
    #       "method_used",
    #        "status",
    #        "tile_path",
    #        "moved_to",
    #    ])

        for row, is_sharp in results:
            writer.writerow(row )
            if is_sharp:
                sharp_count += 1
            else:
                blurry_count += 1
 
    if timed_out_tiles:
        print(f"[blur_fast] {len(timed_out_tiles)} tile(s) timed out and were skipped: "
              f"{timed_out_tiles[:10]}{' ...' if len(timed_out_tiles) > 10 else ''}")
 
    if was_interrupted:
        print(f"[blur_fast] Stopped early by user. Partial results saved to: {csv_path}")
        print(f"[blur_fast] Processed {sharp_count + blurry_count}/{total_tasks} "
              f"before stopping | Sharp: {sharp_count} | Blurry: {blurry_count}")
        raise KeyboardInterrupt("blur_fast interrupted")
 
    print(f"[blur_fast] Finished! CSV saved to: {csv_path}")
    print(f"[blur_fast] Total: {total_tasks} | Sharp: {sharp_count} | "
          f"Blurry: {blurry_count} | Timed out: {len(timed_out_tiles)}")
 


