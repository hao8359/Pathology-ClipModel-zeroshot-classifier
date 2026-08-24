# new file for pipeline building and running summer

import os
import glob
import time
import pandas as pd
import json
import multiprocessing
import numpy as np
import cv2
import pickle
from openslide import OpenSlide, PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from concurrent.futures import ProcessPoolExecutor
import config
from src.preprocess.anonimization import get_anonymous_case_id, get_anon_slide_id, load_mapping, save_case_mapping

def find_svs_files(folder: str):
    """Recursively find all .svs files in a folder."""
    return sorted(glob.glob(os.path.join(folder, "**", "*.svs"), recursive=True))


def process_slide(args):
    slide_path, anon_case_id = args
    slide_name = os.path.basename(slide_path)
    case_id = slide_name[:12] # exctract the substring containing the case id number
    anon_slide_id = slide_name.replace(case_id, anon_case_id)
    #anon_slide_path =
    print(f"\nProcessing slide: {anon_slide_id} for dataset")

    slide_data=[]
    try:
        slide_start = time.time()
        # Open slide to get metadata
        with OpenSlide(slide_path) as slide:
            mask_levels = slide.level_count
            # W, H = slide.level_dimensions[0]
            # Get microns per pixel for X and Y dimensions
            #mpp_x = slide.properties.get(PROPERTY_NAME_MPP_X)
            #mpp_y = slide.properties.get(PROPERTY_NAME_MPP_Y)
            mag = slide.properties.get("aperio.AppMag") # magnification
            mpp = slide.properties.get("openslide.mpp-x") # general microns per pixel

            #print(f"Case ID: {case_id}")
            print(f"Anonymous case ID: {anon_case_id}")
            print(f"Number of levels: {mask_levels}")
            print(f"Magnification X: {mag}")
            print(f"Microns per pixel X: {mpp}")
            #print(f"Heght and width level 0: W = {W}, H = {H}")
            #print(f"Microns per pixel X: {mpp_x}")
            #print(f"Microns per pixel Y: {mpp_y}")
                
            record = {
                #"case_id": case_id,
                "anon_case_id": anon_case_id,
                "anon_slide_name": anon_slide_id,
                #"slide_path": slide_path,
                "staining_type": "H&E",
                "number of levels": mask_levels,
                "max_magnification": mag,
                "mpp": mpp
                }
            slide_data.append(record)
            slide_end = time.time()
            print(f"{slide_name} took {slide_end - slide_start:.2f} seconds")

            return record 

    except Exception as e:
        import traceback
        print(f"Error processing {slide_path}: {e}")
        traceback.print_exc()
        
        return {
        "slide_path": slide_path,
        "error": str(e)
        }

# ADD STAINING TYPE, PATIENT ID 
def slide_metadata():
    svs_folder = config.SINGLE_SVS_FOLDER
    if not svs_folder or not os.path.isdir(svs_folder):
        print("SINGLE_SVS_FOLDER is not a valid folder:", svs_folder)
        return

    svs_files = find_svs_files(svs_folder)
    if not svs_files:
        print("No .svs files found in folder:", svs_folder)
        return
    # directory to save the dataframe
    out_root = config.OUTPUT_DIR_DATA

    if not out_root:
        print("OUTPUT_DIR_DATA is not set in .env/config")
        return
    
    slide_data = []
    #open and get metadata
    for slide_path in svs_files:
        slide_start = time.time()
        slide_name = os.path.basename(slide_path)
        case_id = slide_name[:12] # exctract the substring containing the case id number

        if os.path.exists(config.ANON_MAPPING_FILE):
            with open(config.ANON_MAPPING_FILE, "r") as f:
                case_mapping = json.load(f)
        else:
            case_mapping = {}

        if case_id not in case_mapping:
            new_id = f"CASE_{len(case_mapping)+1:04d}"
            case_mapping[case_id]= new_id

        anon_case_id = case_mapping[case_id] 

        with open(config.ANON_MAPPING_FILE, "w") as f:
            json.dump(case_mapping, f, indent=4)

        print(f"\nProcessing slide: {slide_name} for dataset")

        try:
            # Open slide to get metadata
            
            with OpenSlide(slide_path) as slide:
                mask_levels = slide.level_count
                #W, H = slide.level_dimensions[0]
                # Get microns per pixel for X and Y dimensions
                #mpp_x = slide.properties.get(PROPERTY_NAME_MPP_X)
                #mpp_y = slide.properties.get(PROPERTY_NAME_MPP_Y)
                mag = slide.properties.get("aperio.AppMag") # magnification
                mpp = slide.properties.get("openslide.mpp-x") # general microns per pixel

                print(f"Case ID: {case_id}")
                print(f"Number of levels: {mask_levels}")
                print(f"Magnification X: {mag}")
                print(f"Microns per pixel X: {mpp}")
                #print(f"Heght and width level 0: W = {W}, H = {H}")
                #print(f"Microns per pixel X: {mpp_x}")
                #print(f"Microns per pixel Y: {mpp_y}")
                
                record = {
                    "case_id": case_id,
                    "anon_case_id": anon_case_id,
                    "slide_name":slide_name,
                    "slide_path": slide_path,
                    "staining_type": "H&E",
                    "number of levels": mask_levels,
                    "max_magnification": mag,
                    "mpp": mpp
                    }
                slide_data.append(record)

        except Exception as e:
            print(f"Error opening {slide_path}: {e}")
        
        df = pd.DataFrame(slide_data)

        #if the directory does not exist, create it
        os.makedirs(out_root, exist_ok=True)
        slide_output = os.path.join(out_root, "slides_metadata.csv")
        slide_output_pkl = os.path.join(out_root, "slides_metadata.pkl")
        df.to_csv(slide_output, index=False)
        df.to_pickle(slide_output_pkl)

        slide_end = time.time()
        print(f"{slide_name} took {slide_end - slide_start:.2f} seconds")

        print(f"Dataset summary saved to: {slide_output} and {slide_output_pkl}")   

def slide_metadata1():
    svs_folder = config.SINGLE_SVS_FOLDER
    if not svs_folder or not os.path.isdir(svs_folder):
        print("SINGLE_SVS_FOLDER is not a valid folder:", svs_folder)
        return

    svs_files = find_svs_files(svs_folder)
    if not svs_files:
        print("No .svs files found in folder:", svs_folder)
        return
    # directory to save the dataframe
    out_root = config.OUTPUT_DIR_DATA

    if not out_root:
        print("OUTPUT_DIR_DATA is not set in .env/config")
        return
    
    # create the output directory immediately, before anything else can fail
    os.makedirs(os.path.dirname(config.ANON_MAPPING_FILE), exist_ok=True)
    
    # create anonymization before running parallel process
    slide_info = []
    case_mapping = load_mapping()

    for slide_path in svs_files:
        slide_name = os.path.basename(slide_path)
        case_id = slide_name[:12]
        
        anon_case_id = get_anonymous_case_id(case_id, case_mapping)
        slide_info.append((slide_path, anon_case_id))
    
    save_case_mapping(case_mapping)

    dataset_start = time.time()
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    # open and get metadata
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(process_slide, slide_info)
        
    results = [r for r in results if r is not None]
      
    df = pd.DataFrame(results)

    #if the directory does not exist, create it
    os.makedirs(out_root, exist_ok=True)
    slide_output = os.path.join(out_root, "slides_metadata.csv")
    slide_output_pkl = os.path.join(out_root, "slides_metadata.pkl")
    df.to_csv(slide_output, index=False)
    df.to_pickle(slide_output_pkl)

    print(f"Parallel runtime: {time.time() - dataset_start:.2f} seconds")

    print(f"Dataset summary saved to: {slide_output} and {slide_output_pkl}")         



if __name__ == "__main__":

    slide_metadata1()     

