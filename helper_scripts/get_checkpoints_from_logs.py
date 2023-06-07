import argparse
import json
import os.path
import re
import shutil
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

from utils.model_utils import get_readable_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_checkpoint_props(checkpoint_path: str) -> Dict[str, Any]:
    """
    Given name of checkpoint path, extracts relevant properties from string

    :param checkpoint_path: Path or basename of model checkpoint to be analyzed
    :return: Dict of checkpoint properties, val loss, train loss, and epoch
    """
    props = {}

    val_index = checkpoint_path.index("val")
    val_loss = float(checkpoint_path[val_index + 9:val_index + 15])
    props["val_loss"] = val_loss

    try:
        train_index = checkpoint_path.index("train")
        train_loss = float(checkpoint_path[train_index + 11:train_index + 17])
        props["train_loss"] = train_loss
    except ValueError:
        props["train_loss"] = 999

    epoch_index = checkpoint_path.index("epoch")
    epoch = int(checkpoint_path[epoch_index + 6:epoch_index + 9])
    props["epoch"] = epoch

    # get checkpoint time string
    time_search = re.compile(".*(\d\d\d\d:\d\d:\d\d:\d\d:\d\d:\d\d).hdf5")
    time_str = time_search.search(checkpoint_path).group(1)
    props["checkpoint_time_str"] = time_str

    # get model name
    name_search = re.compile("model-(.*)_seq-.*")
    model_name = name_search.search(checkpoint_path).group(1)
    props["model_name"] = model_name

    return props


def get_best_checkpoint(candidate_jsons: List[Dict[str, Any]], checkpoint_dir: str, criteria_key: str = "val"):
    assert criteria_key == "val" or criteria_key == "train", "only val and train supported"
    best_props = None
    best_cand_value = float("inf")
    for candidate in candidate_jsons:
        cand_value = candidate[f"best_{criteria_key}_loss"]
        if cand_value < best_cand_value:
            best_props = {
                f"{criteria_key}_loss": round(cand_value, 4),
                "epoch": candidate[f"best_{criteria_key}_epoch"] + 1,  # checkpoints epoch 1 indexed, jsons 0-indexed
                "model_name": get_readable_name(candidate["model_params"])
            }
            if "checkpoint_time_str" in candidate:
                best_props["checkpoint_time_str"] = candidate["checkpoint_time_str"]

    for checkpoint in os.listdir(checkpoint_dir):
        if ".hdf5" not in checkpoint:
            continue
        props = get_checkpoint_props(checkpoint)
        if best_props.items() <= props.items():
            return os.path.join(checkpoint_dir, checkpoint)

    raise ValueError(f"No checkpoint matching props in json {best_props} found")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_json_list(json_dir: str, checkpoint_dir: str, out_dir: str):
    json_map = defaultdict(list)
    # separate jsons by class
    re_match = re.compile("(?:hyperparam_tuning_)?(.*)_\d_train_results.json")
    for file in os.listdir(json_dir):
        match = re_match.search(file)
        if match is not None:
            model_type = match.group(1)
            # read json data and save
            json_path = os.path.join(json_dir, file)
            try:
                parsed = read_json(json_path)
                json_map[model_type].append(parsed)
            except JSONDecodeError:
                print(f"Could not parse json at {json_path}, skipping")
                continue

    for candidate in ["val", "train"]:
        params_map = {}
        # for each class, get best checkpoint
        dest = os.path.join(out_dir, candidate)
        Path(dest).mkdir(exist_ok=True, parents=True)
        for model_type, json_data in json_map.items():
            checkpoint_path = get_best_checkpoint(candidate_jsons=json_data, checkpoint_dir=checkpoint_dir,
                                                  criteria_key=candidate)
            shutil.copy(checkpoint_path, dest)
            params_map[os.path.basename(checkpoint_path)] = json_data[0]["model_params"]

        with open(os.path.join(dest, "params.json"), "w") as f:
            json.dump(params_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str)
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="out_models")
    args = parser.parse_args()
    process_json_list(args.json_dir, args.checkpoint_dir, args.out_dir)
