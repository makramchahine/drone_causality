import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.vis_utils import ARROW_BOX_HEIGHT, show_vel_cmd


def visualize_run(run_dir: str, csv_path: str, output_path: str):
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    imgs = sorted(os.listdir(run_dir))
    imgs = [os.path.join(run_dir, img) for img in imgs if "png" in img]
    frame_size = list(cv2.imread(imgs[0]).shape[:2])
    frame_size[0] += ARROW_BOX_HEIGHT

    # videowriter takes width, height, image_shape is height, width
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, frame_size[::-1],
                             True)  # true means write color frames
    csv_dat = pd.read_csv(csv_path)

    for i, img_path in tqdm(enumerate(imgs)):
        img = cv2.imread(img_path)
        vel_cmd = csv_dat.iloc[i].to_numpy()
        vel_cmd = np.expand_dims(vel_cmd, axis=0)
        command = show_vel_cmd(vel_cmd, frame_size[1])
        stacked = np.concatenate((img, command), axis=0)
        writer.write(stacked)

    writer.release()


def visualize_processed_run(run_dir: str, output_path: str):
    for run in os.listdir(run_dir):
        run = os.path.join(run_dir, run)
        csv_path = os.path.join(run, "data_out.csv")
        run_output = os.path.join(output_path, f"{run}.mp4")
        visualize_run(run, csv_path, run_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    # parser.add_argument("csv_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    # visualize_run(args.run_dir, args.csv_path, args.output_path)
    visualize_processed_run(run_dir=args.run_dir, output_path=args.output_path)
