# Created by Patrick Kao at 4/18/22
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Sequence


def merge_model_dirs(merge_dirs: Sequence[str], out_dir: str):
    for model_type in ["train", "val"]:
        out_json = {}
        for model_dir in merge_dirs:
            dir_path = os.path.join(model_dir, model_type)
            type_out = os.path.join(out_dir, model_type)
            Path(type_out).mkdir(parents=True, exist_ok=True)
            contents = os.listdir(dir_path)
            model_names = [file for file in contents if ".hdf5" in file]
            for model in model_names:
                abs_path = os.path.join(dir_path, model)
                shutil.copy(abs_path, type_out)

            with open(os.path.join(dir_path, "params.json"), "r") as f:
                param_data = json.load(f)

            out_json.update(param_data)

            with open(os.path.join(type_out, "params.json"), "w") as f:
                json.dump(out_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("merge_dirs", nargs='+', default=[])
    parser.add_argument("--out_dir", default="merged_models")
    args = parser.parse_args()
    merge_model_dirs(args.merge_dirs, args.out_dir)
