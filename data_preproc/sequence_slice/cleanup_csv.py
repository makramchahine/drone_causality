import os
import numpy as np
import pandas as pd
from transformations import euler_from_quaternion

path = '/home/ramin/devens_drone_data/devens_12102021_sliced/'

L = os.listdir(path)

while True:
    for flight in L:
        # if flight != '__pychache__':
        df = pd.read_csv(path + flight + '/data_out.csv')
        yaws = np.zeros(len(df))
        try:
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

            df_training.to_csv(path + flight + '/data_out.csv', index=False)
            break

        except:
            print(flight+' already processed')
