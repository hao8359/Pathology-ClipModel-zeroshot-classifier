import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(data_dir):
    # 1. Search recursively for data.json inside data_dir
    json_path = None
    base_dir = None
    for root, _, files in os.walk(data_dir):
        if "data.json" in files:
            json_path = os.path.join(root, "data.json")
            base_dir = root
            break

    if json_path is None:
        raise FileNotFoundError(f"Could not find 'data.json' inside: {data_dir}")

    print(f"Found metadata file at: {json_path}")

    # 2. Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")

    # Verify expected keys
    if "img" not in df.columns or "caption" not in df.columns:
        raise KeyError(
            f"Expected 'img' and 'caption' keys in data.json, but found: {list(df.columns)}"
        )

    # Rename 'img' -> 'filepath' for Open_CLIP
    df = df.rename(columns={"img": "filepath"})

    # 3. Resolve image paths (check if images/ subfolder exists)
    images_subfolder = os.path.join(base_dir, "images")
    image_dir = images_subfolder if os.path.exists(images_subfolder) else base_dir

    print(f"Resolving image files from directory: {image_dir}")
    df["filepath"] = df["filepath"].apply(lambda x: os.path.join(image_dir, str(x)))

    # 4. Split dataset into 90% Training and 10% Validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # 5. Export TSV files to data_dir root
    train_tsv = os.path.join(data_dir, "pathcap_train.tsv")
    val_tsv = os.path.join(data_dir, "pathcap_val.tsv")

    train_df[["filepath", "caption"]].to_csv(train_tsv, sep="\t", index=False)
    val_df[["filepath", "caption"]].to_csv(val_tsv, sep="\t", index=False)

    print(f"Dataset preparation complete:")
    print(f" - Train samples: {len(train_df)} -> {train_tsv}")
    print(f" - Val samples:   {len(val_df)} -> {val_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    prepare_dataset(args.data_dir)