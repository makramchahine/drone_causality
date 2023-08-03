import argparse
import os
import shutil
import sys
from pathlib import Path

import PIL
import PIL.ImageOps
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from pandas import DataFrame
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from sequence_slice.transformations import euler_from_quaternion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from keras_models import IMAGE_SHAPE

normalize = True
CSV_NAME = "data_out_base.csv" if normalize else "data_out.csv"
CSV_NAME_2 = "data_in.csv"
POS_CSV = "pos.csv"

def process_csv(df: DataFrame, out_dir: str) -> DataFrame:
    """
    Applies relative heading transformation to csv file of sensor readings and saves relevant cols for training

    :param df: pandas df containing sensor readings collected during logging
    :return: Transformed dataframe
    """
    yaws = np.array(df['yaw'])

    vx_body = df.vx * np.cos(-yaws) - df.vy * np.sin(-yaws)
    vy_body = df.vx * np.sin(-yaws) + df.vy * np.cos(-yaws)
    df_training = pd.DataFrame()
    df_training['vx'] = vx_body
    df_training['vy'] = vy_body
    df_training['vz'] = df.vz
    df_training['omega_z'] = df.yaw_rate

    # for values of omega_z that are larger than 2 or smaller than -2, set them to the value preceding them
    for i in range(len(df_training['omega_z'])):
        if df_training['omega_z'][i] > 2 or df_training['omega_z'][i] < -2:
            if i == 0:
                df_training['omega_z'][i] = 0
            else:
                df_training['omega_z'][i] = df_training['omega_z'][i-1]

    df_training['time_total'] = df.time_total
    df_training = df_training.drop_duplicates(subset='time_total', keep="first")

    df_training = df_training.drop('time_total', axis=1)

    #add header to df with column names vx, vy, vz, omega_z
    df_training.columns = ['vx', 'vy', 'vz', 'omega_z']


    return df_training

def process_csv_pos(df: DataFrame, out_dir: str) -> DataFrame:
    """
    Applies relative heading transformation to csv file of sensor readings and saves relevant cols for training

    :param df: pandas df containing sensor readings collected during logging
    :return: Transformed dataframe
    """
    df_training = pd.DataFrame()
    df_training['x'] = df.x
    df_training['y'] = df.y
    df_training['z'] = df.z
    df_training['yaw'] = df.yaw

    df_training['time_total'] = df.time_total
    df_training = df_training.drop_duplicates(subset='time_total', keep="first")

    df_training = df_training.drop('time_total', axis=1)

    #add header to df with column names x, y, z, yaw
    df_training.columns = ['x', 'y', 'z', 'yaw']

    return df_training


def process_image(img: Image.Image, flip_channels: bool = True) -> Image.Image:
    """
    Applies image transformations to training Data. Resizes to IMAGE_SIZE
    :param img:
    :return:
    """
    desired_shape = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
    if img.size != desired_shape:
        img = img.resize(desired_shape, resample=PIL.Image.BICUBIC)
    if flip_channels:
        # noinspection PyTypeChecker
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(img)

    return img


def process_data(data_dir: str, out_dir: str, flip_channels: bool = False) -> None:
    """
    Processes all runs collected in the session by data_dir and saves to out_dir
    """
    dirs = os.listdir(data_dir)
    dirs = sorted([d for d in dirs if ('csv' not in d and 'mp4' not in d)])

    def process_one_run(run_dir: str):
        run_abs = os.path.join(data_dir, run_dir)
        run_out_dir = run_dir
        Path(os.path.join(out_dir, run_out_dir)).mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(os.path.join(run_abs, 'log_0.csv'), header=0)
            df_training = process_csv(df, os.path.join(out_dir, run_out_dir))
            # df = pd.read_csv(os.path.join(run_abs, 'log_1.csv'), header=0)
            # df_training2 = process_csv(df, os.path.join(out_dir, run_out_dir))
            df_training_pos = process_csv_pos(df, os.path.join(out_dir, run_out_dir))
            #skip first row
            df_training = df_training[1:]
            # duplicate all columns
            # df_training = pd.concat([df_training, df_training2], axis=1, ignore_index=False)
            df_training.to_csv(os.path.join(out_dir, run_out_dir, CSV_NAME), index=False)
            df_training_pos = df_training_pos[1:]
            df_training_pos.to_csv(os.path.join(out_dir, run_out_dir, POS_CSV), index=False)

            df = pd.read_csv(os.path.join(run_abs, 'values.csv'), header=None)
            df.columns = ['direction']
            # create a new df with 2
            # for each line in df, if the value is equal to 1 write a line containing the values 1 and 0
            # if the value is equal to -1 write a line containing the values 0 and 1
            nu_df = pd.DataFrame(columns=['L', 'R'])
            for index, row in df.iterrows():
                if row['direction'] == 1:
                    nu_df.loc[index] = [1, 0]
                elif row['direction'] == -1:
                    nu_df.loc[index] = [0, 1]
                else:
                    raise ValueError("Invalid value in direction column")
            nu_df = nu_df[1:]
            nu_df.to_csv(os.path.join(out_dir, run_out_dir, CSV_NAME_2), index=False)

        except FileNotFoundError:
            print(f"Could not find csv for run {run_dir}. Assuming already processed and copying existing csv")
            shutil.copy(os.path.join(run_abs, CSV_NAME), os.path.join(out_dir, run_out_dir, CSV_NAME))

        # for d, letter in zip(range(2), ["a", "b"]):
        for d, letter in zip(range(1), ["a"]):
            img_files = sorted(os.listdir(run_abs+f"/pics{d}"))
            img_files = [os.path.join(run_abs, f"pics{d}", img) for img in img_files if "png" in img]
            img_files = img_files[1:]


            for (ix, im_path) in enumerate(img_files):
                # f format string for ix with 6 digits:
                img_out_path = os.path.join(out_dir, run_out_dir, f'{ix:06d}{letter}.png')
                if os.path.exists(img_out_path):
                    continue
                img = Image.open(im_path)
                im_smaller = process_image(img, flip_channels)
                im_smaller.save(img_out_path)
    # for run_dir in tqdm(dirs):
    #     process_one_run(run_dir)
    # run image processing in different threads
    Parallel(n_jobs=16)(delayed(process_one_run)(run_dir) for run_dir in tqdm(dirs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="absolute path of data to be processed")
    parser.add_argument("out_dir", type=str, help="absolute path to output location for processed data")
    parser.add_argument("--flip_channels", action="store_true")
    args = parser.parse_args()
    process_data(args.data_dir, args.out_dir, flip_channels=args.flip_channels)
