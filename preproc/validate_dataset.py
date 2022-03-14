# make sure csv and img dir have same num rows
import argparse
import itertools
import os
import sys
from typing import Any, Dict

import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from keras_models import IMAGE_SHAPE

DESIRED_SHAPE = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


def file_to_int(img_path: str) -> int:
    return int(os.path.splitext(os.path.basename(img_path))[0])


def validate_run(run_path: str, processed):
    to_ret = []
    # load csv
    try:
        run_num = os.path.basename(os.path.dirname(run_path))
        csv_filename = os.path.join(os.path.dirname(run_path), "..", '%.2f.csv' % float(run_num))
        control_df = pd.read_csv(csv_filename)
    except (FileNotFoundError, ValueError) as e:
        csv_filename = os.path.join(run_path, "data_out.csv")
        control_df = pd.read_csv(csv_filename)

    # check num imgs = len of csv
    imgs = os.listdir(run_path)
    imgs = [os.path.join(run_path, img) for img in imgs if "png" in img]
    imgs.sort(key=file_to_int)
    num_imgs = len(imgs)
    if num_imgs != control_df.shape[0]:
        err_str = f"path {run_path} has {num_imgs} imgs while csv has {control_df.shape[0]} rows"
        to_ret.append((run_path, err_str))
        print(err_str)

    # try to load each image
    for i, img_path in enumerate(imgs):
        try:
            img = Image.open(img_path)
            if processed:
                img_idx_name = file_to_int(img_path)
                if img_idx_name != i:
                    err_str = f"{img_path} has name {img_idx_name}, should have name {i}"
                    to_ret.append(err_str)
                    print(err_str)
                if img.size != DESIRED_SHAPE:
                    err_str = f"{img_path} has size {img.size}, should have size {DESIRED_SHAPE}"
                    to_ret.append(err_str)
                    print(err_str)
        except UnidentifiedImageError:
            err_str = f"Could not load image {img_path}"
            to_ret.append((run_path, err_str, img_path))
            print(err_str)

    return to_ret


def validate_dataset(dataset_dir: str, processed: bool = True) -> Dict[str, Any]:
    run_data = sorted(os.listdir(dataset_dir))
    run_data = [os.path.join(dataset_dir, img) for img in run_data]
    results = Parallel(n_jobs=6)(delayed(validate_run)(run_path, processed) for run_path in tqdm(run_data))
    results = itertools.chain(*results)
    bad = {}
    for res in results:
        if len(res) > 0:
            bad[res[0]] = res[1:]

    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    bad = validate_dataset(args.dataset_dir)
    print(bad)
