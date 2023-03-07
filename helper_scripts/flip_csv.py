import os
from typing import Sequence

import pandas as pd


def flip_csv(csv_file: str, columns: Sequence[str]):
    df = pd.read_csv(csv_file)
    for col in columns:
        df[col] = df[col].apply(lambda x: x*-1)

    df.to_csv(csv_file, index=False)

data = "/home/dolphonie/Desktop/mixed_aug_fixed"
for folder in os.listdir(data):
    flip_csv(os.path.join(data, folder, "data_out.csv"), ["vz", "omega_z"])