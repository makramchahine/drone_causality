import argparse
import os.path
from datetime import datetime
from os import listdir
from os.path import isfile
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_checkpoint_epoch(checkpoint_dir: str, epoch: int, json_time: float):
    # get mapping of time to checkpoint path
    checkpoint_to_time = {}
    for child in listdir(checkpoint_dir):
        # filter only model checkpoints
        if isfile(os.path.join(checkpoint_dir, child)) and "hdf5" in child:
            name, extension = os.path.splitext(child)
            time_str = "_".split(name)[-1]
            dt = datetime.strptime(time_str, "%Y:%m:%d:%H:%M:%S")
            time_sec = dt.timestamp()
            checkpoint_to_time[child] = time_sec

    # get checkpoints before and with epoch
    legal_map = {}
    for checkpoint, time in checkpoint_to_time.items():
        if f"epoch-{epoch:03d}" in checkpoint and time < json_time:
            legal_map[checkpoint] = time

    # get closest checkpoint in time
    max_checkpoint = max(legal_map, key=legal_map.get)
    return max_checkpoint


def get_best_val(checkpoint_dir: str, filters: List[str]):
    checkpoint_dir = os.path.join(SCRIPT_DIR, checkpoint_dir)

    checkpoint_to_loss = {}
    for child in listdir(checkpoint_dir):
        # filter only jsons
        if isfile(os.path.join(checkpoint_dir, child)) and "hdf5" in child:
            # make sure all filters in child
            satisfies_filters = True
            for filter_str in filters:
                satisfies_filters = satisfies_filters and filter_str in child
            if not satisfies_filters:
                continue

            try:
                val_index = child.index("val")
            except ValueError:
                print(f"substring val not found in {child}")
                continue
            val_loss = float(child[val_index + 9:val_index + 15])
            checkpoint_to_loss[child] = val_loss

    best_checkpoint = min(checkpoint_to_loss, key=checkpoint_to_loss.get)
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("filter_str", nargs='+', type=str)
    args = parser.parse_args()
    print(get_best_val(args.checkpoint_dir, args.filter_str))
