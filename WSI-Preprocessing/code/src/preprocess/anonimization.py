import json
import os
import config
import uuid

#load existing json mapping
def load_mapping():

    os.makedirs(os.path.dirname(config.ANON_MAPPING_FILE), exist_ok=True)
    if os.path.exists(config.ANON_MAPPING_FILE):
        with open(config.ANON_MAPPING_FILE, "r") as f:
            case_mapping = json.load(f)
    else:
        case_mapping = {}
    return case_mapping

# create new anonymous case ids
def get_anonymous_case_id(case_id, case_mapping):
    
    #case_mapping = case_mapping or config.ANON_MAPPING_FILE
    
    if case_id not in case_mapping:
        new_id = str(uuid.uuid4())
        case_mapping[case_id]= new_id

    anon_case_id = case_mapping[case_id] 
    return anon_case_id

# save the mapping
def save_case_mapping(case_mapping):

    os.makedirs(os.path.dirname(config.ANON_MAPPING_FILE), exist_ok=True)
    with open(config.ANON_MAPPING_FILE, "w") as f:
        json.dump(case_mapping, f, indent=4)

# uses the mapping to make an anonimized directory name -- not compatible with pipeline
def directory_mapping(case_id, output_dir, case_mapping_file):

    case_mapping_file = case_mapping_file or config.ANON_MAPPING_FILE

    if not os.path.exists(case_mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {case_mapping_file}")

    with open(case_mapping_file, "r") as f:
        case_mapping = json.load(f)
    if case_id not in case_mapping:
        raise KeyError (f"Case ID {case_id} is not anonymized")
    anon_case_id = case_mapping[case_id]

    return os.path.join(output_dir, anon_case_id)


def get_anon_slide_id(slide_name, case_mapping):
    """
    Given a slide filename, return an anonymized slide_id:
    anon_case_id + whatever came after the case_id in the original name.
    Looks up the mapping only, does not create new entries.
    """
    slide_id, ext = os.path.splitext(slide_name)
    case_id = slide_name[:12]
    remainder = slide_id[12:]  # e.g. "-01Z-00-DX1"

    if case_id not in case_mapping:
        raise KeyError(f"No anonymized ID found for case_id: {case_id}")

    return case_mapping[case_id] + remainder + ext
    
