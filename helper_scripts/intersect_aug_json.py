# Created by Patrick Kao at 3/16/22
"""
Calculates the intersection of the img dirs of the images in a data processing json file and a data directory and
only saves the json with corresponding entries in the data directory
"""
import argparse
import json
import os.path
from typing import Any, Dict


def get_intersection_json(data_json: str, data_dir: str, out_path: str = "intersect.json") -> Dict[str, Any]:
    to_ret = []
    with open(data_json, "r") as f:
        synth_data = json.load(f)

    for img_path, center_coords in synth_data:
        img_dir = os.path.basename(os.path.dirname(img_path))
        if os.path.exists(os.path.join(data_dir, img_dir)):
            to_ret.append([img_path, center_coords])

    with open(out_path, "w") as f:
        json.dump(to_ret, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", type=str)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    get_intersection_json(args.data_json, args.data_dir)
