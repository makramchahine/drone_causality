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

    epoch_search = re.compile("epoch-(\d+)")
    epoch = int(epoch_search.search(checkpoint_path).group(1))
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
    print(candidate_jsons)
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
        best_props.pop("epoch", None)
        if best_props.items() <= props.items():
            print("returned: ", os.path.join(checkpoint_dir, checkpoint))
            return os.path.join(checkpoint_dir, checkpoint), best_props["checkpoint_time_str"]

    raise ValueError(f"No checkpoint matching props in json {best_props} found")

def get_spaced_out_checkpoints(candidate_jsons: List[Dict[str, Any]], checkpoint_dir: str, frequency: int, val_best_prop_timestamp: str):
    checkpoint_paths = []
    checkpoint_epochs = []
    for candidate in candidate_jsons:
        checkpoint_time_str = candidate["checkpoint_time_str"]
        if checkpoint_time_str != val_best_prop_timestamp:
            continue
        for checkpoint in os.listdir(checkpoint_dir):
            if ".hdf5" not in checkpoint:
                continue
            props = get_checkpoint_props(checkpoint)
            if checkpoint_time_str == props["checkpoint_time_str"] and props["epoch"] % frequency == 0:
                checkpoint_paths.append(os.path.join(checkpoint_dir, checkpoint))
                checkpoint_epochs.append(props["epoch"])

    return checkpoint_paths, checkpoint_epochs


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_json_list(json_dir: str, checkpoint_dir: str, out_dir: str):
    json_map = defaultdict(list)
    loss_map = defaultdict(dict)
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
                loss_map[model_type]["loss"] = parsed["loss"]
                loss_map[model_type]["val_loss"] = parsed["val_loss"]
            except JSONDecodeError:
                print(f"Could not parse json at {json_path}, skipping")
                continue

    print(f"json_map: {json_map}")
    for candidate in ["val", "train"]:
        params_map = {}
        # for each class, get best checkpoint
        dest = os.path.join(out_dir, candidate)
        Path(dest).mkdir(exist_ok=True, parents=True)
        for model_type, json_data in json_map.items():
            checkpoint_path, best_prop_timestamp = get_best_checkpoint(candidate_jsons=json_data, checkpoint_dir=checkpoint_dir,
                                                  criteria_key=candidate)
            shutil.copy(checkpoint_path, dest)
            print(f"copied {checkpoint_path} to {dest}" )
            params_map[os.path.basename(checkpoint_path)] = json_data[0]["model_params"]

            if candidate == "val":
                val_best_prop_timestamp = best_prop_timestamp

        with open(os.path.join(dest, "params.json"), "w") as f:
            json.dump(params_map, f)

    recurring_checkpoint_paths, checkpoint_epochs = get_spaced_out_checkpoints(candidate_jsons=json_data, checkpoint_dir=checkpoint_dir, frequency=50, val_best_prop_timestamp=val_best_prop_timestamp)
    dest = os.path.join(out_dir, "recurrent")
    Path(dest).mkdir(exist_ok=True, parents=True)
    for checkpoint_path, checkpoint_epoch in zip(recurring_checkpoint_paths, checkpoint_epochs):
        shutil.copy(checkpoint_path, dest)
        # print(f"copied {checkpoint_path} to {dest}")
        params_map = {}
        params_map[os.path.basename(checkpoint_path)] = json_data[0]["model_params"]
        with open(os.path.join(dest, f"params{checkpoint_epoch}.json"), "w") as f:
            json.dump(params_map, f)

    with open(os.path.join(out_dir, "loss.json"), "w") as f:
        json.dump(loss_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str)
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="out_models")
    args = parser.parse_args()
    process_json_list(args.json_dir, args.checkpoint_dir, args.out_dir)
