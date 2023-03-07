# Created by Patrick Kao at 5/3/22
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from keras_models import IMAGE_SHAPE
from utils.data_utils import load_image


def flip_channels(im_dir: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for im_path in os.listdir(im_dir):
        img = load_image(os.path.join(im_dir, im_path), IMAGE_SHAPE, reverse_channels=False) # writing flips channels
        cv2.imwrite(os.path.join(out_dir, im_path), np.squeeze(img, axis=0), )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("im_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    flip_channels(args.im_dir, args.out_dir)