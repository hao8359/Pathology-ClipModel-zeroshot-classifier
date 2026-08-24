#temp_file.py

import tempfile
import shutil
import os 
import glob
from src.preprocess.anonimization import load_mapping, get_anon_slide_id
from contextlib import contextmanager
import logging
logger = logging.getLogger(__name__)

def find_svs_files(folder: str):
    """Recursively find all .svs files in a folder."""
    return sorted(glob.glob(os.path.join(folder, "**", "*.svs"), recursive=True))

def create_temporary_copy(slide_path):
    temp_dir = tempfile.gettempdir()
    slide_name = os.path.basename(slide_path)
    case_mapping = load_mapping()
    try:
        temp_file_name = get_anon_slide_id(slide_name, case_mapping)
    except KeyError as e:
        logger.warning(f"Skipping {slide_name}: {e}")
    temp_path = os.path.join(temp_dir, temp_file_name)
    shutil.copy2(slide_path, temp_path)
    return temp_path


@contextmanager
def anonymized_batch(slide_paths):
    case_mapping = load_mapping()          # load once
    work_dir = tempfile.mkdtemp(prefix="anon_")  # isolated per-run dir
    anon_paths = []
    try:
        for anon_path in slide_paths:
            slide_name = os.path.basename(anon_path)
            temp_file_name = get_anon_slide_id(slide_name, case_mapping)
            temp_path = os.path.join(work_dir, temp_file_name)
            shutil.copy2(anon_path, temp_path)
            anon_paths.append(temp_path)
        yield anon_paths
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
	
@contextmanager
def anonymized_slide(slide_path, work_dir, case_mapping):
    """Stage exactly one slide into work_dir; clean it up on exit."""
    slide_name = os.path.basename(slide_path)
    temp_file_name = get_anon_slide_id(slide_name, case_mapping)
    temp_path = os.path.join(work_dir, temp_file_name)
    shutil.copy2(slide_path, temp_path)
    try:
        yield temp_path
    finally:
        # free scratch space immediately once this slide's tiling is done
        if os.path.exists(temp_path):
            os.remove(temp_path)
