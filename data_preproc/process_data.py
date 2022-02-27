import argparse
import os
import sys
from pathlib import Path

import PIL
import PIL.ImageOps
import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from tqdm import tqdm

from sequence_slice.transformations import euler_from_quaternion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from keras_models import IMAGE_SHAPE


# bad_runs = []
# plt.ion()
# for d in dirs:
#    df = pd.read_csv(os.path.join('raw_data', '%.2f.csv' % float(d)))
#    plt.figure()
#    plt.plot(df['lat'], df['lng'])
#    plt.xlim([.00035 + 42.521, .00065 + 42.521])
#    plt.ylim([-.0006 - 71.606, -.00035 - 71.606])
#
#    keep = raw_input('Data ok? [Y/n]')
#    plt.close()
#    keep = keep == '' or keep == 'Y' or keep == 'y'
#
#    if not keep:
#        bad_runs.append(d)

def process_csv(df: DataFrame) -> DataFrame:
    """
    Applies relative heading transformation to csv file of sensor readings and saves relevant cols for training

    :param df: pandas df containing sensor readings collected during logging
    :return: Transformed dataframe
    """
    yaws = np.zeros(len(df))
    for ix in range(len(yaws)):
        quat = [df['att_x'][ix], df['att_y'][ix], df['att_z'][ix], df['att_w'][ix]]
        yaws[ix] = euler_from_quaternion(quat)[2]

    vx_body = df.vx * np.cos(-yaws) - df.vy * np.sin(-yaws)
    vy_body = df.vx * np.sin(-yaws) + df.vy * np.cos(-yaws)
    df_training = pd.DataFrame()
    df_training['vx'] = vx_body
    df_training['vy'] = vy_body
    df_training['vz'] = df.vz
    df_training['omega_z'] = df.ang_vel_z
    return df_training


def process_image(img: Image.Image) -> Image.Image:
    """
    Applies image transformations to training Data. Resizes to IMAGE_SIZE
    :param img:
    :return:
    """
    return Image.fromarray(
        np.array(img.resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0]), resample=PIL.Image.BICUBIC))[:, :, ::-1])


def process_data(data_dir: str, out_dir: str) -> None:
    """
    Processes all runs collected in the session by data_dir and saves to out_dir
    """
    dirs = os.listdir(data_dir)
    dirs = sorted([d for d in dirs if 'csv' not in d])

    for (run_ix, run_dir) in tqdm(enumerate(dirs)):
        run_abs = os.path.join(data_dir, run_dir)
        run_out_dir = '%.2f' % float(run_dir)

        df = pd.read_csv(os.path.join(data_dir, '%.2f.csv' % float(run_dir)))

        df_training = process_csv(df)

        Path(os.path.join(out_dir, run_out_dir)).mkdir(parents=True, exist_ok=True)
        df_training.to_csv(os.path.join(out_dir, run_out_dir, 'data_out.csv'), index=False)

        img_files = sorted(os.listdir(run_abs))

        for (ix, fn) in enumerate(img_files):
            im = Image.open(os.path.join(run_abs, fn))
            im_smaller = process_image(im)
            im_smaller.save(os.path.join(out_dir, run_out_dir, '%06d.png' % ix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="absolute path of data to be processed")
    parser.add_argument("out_dir", type=str, help="absolute path to output location for processed data")
    args = parser.parse_args()
    process_data(args.data_dir, args.out_dir)
