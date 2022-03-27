import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.vis_utils import ARROW_BOX_HEIGHT, show_vel_cmd


def visualize_run(run_dir: str, output_path: str, csv_path: Optional[str] = None):
    if csv_path is None:
        csv_path = os.path.join(run_dir, "data_out.csv")
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    imgs = sorted(os.listdir(run_dir))
    imgs = [os.path.join(run_dir, img) for img in imgs if "png" in img]
    frame_size = list(cv2.imread(imgs[0]).shape[:2])
    frame_size[0] += ARROW_BOX_HEIGHT

    # videowriter takes width, height, image_shape is height, width
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, frame_size[::-1],
                             True)  # true means write color frames
    csv_dat = pd.read_csv(csv_path)

    for i, img_path in tqdm(enumerate(imgs)):
        if i >= csv_dat.shape[0]:
            break
        img = cv2.imread(img_path)
        vel_cmd = csv_dat.iloc[i][["cmd_vx", "cmd_vy", "cmd_vz", "cmd_omega"]].to_numpy()
        vel_cmd = np.expand_dims(vel_cmd, axis=0)
        command = show_vel_cmd(vel_cmd, frame_size[1])
        stacked = np.concatenate((img, command), axis=0)
        writer.write(stacked)

    writer.release()


def visualize_processed_runs(run_dir: str, output_path: str):
    for run in os.listdir(run_dir):
        run = os.path.join(run_dir, run)
        run_output = os.path.join(output_path, f"{run}.mp4")
        visualize_run(run, run_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()
    visualize_run(args.run_dir, args.output_path, csv_path=args.csv_path)
    # visualize_processed_runs(run_dir=args.run_dir, output_path=args.output_path)
