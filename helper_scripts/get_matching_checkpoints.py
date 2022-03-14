# Created by Patrick Kao at 3/11/22
import argparse
import json
import os
import shutil
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_matching_checkpoints(checkpoint_dir: str, filter_str: str, params_str: str,
                             out_dir: str = "matching_checkpoints"):
    """
    Finds all checkpoints matching filter_str and creates a params.json file for them with the params_str given by
    params_str
    :return:
    """

    model_params = {}
    out_dir = os.path.join(SCRIPT_DIR, out_dir)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    for checkpoint in sorted(os.listdir(checkpoint_dir)):
        if filter_str in checkpoint and ".hdf5" in checkpoint:
            model_params[checkpoint] = params_str
            shutil.copy(os.path.join(checkpoint_dir, checkpoint), out_dir)

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(model_params, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("filter_str", type=str)
    parser.add_argument("params_str", type=str)
    args = parser.parse_args()
    get_matching_checkpoints(args.checkpoint_dir, args.filter_str, args.params_str)
