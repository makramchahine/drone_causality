#!/usr/bin/python3
import pandas as pd
import os
import numpy as np
from tf.transformations import euler_from_quaternion

training_inputs_fn = 'training_inputs_w_attitude.csv'
dirs = os.listdir('.')
for (ix, d) in enumerate(dirs):
    print('Processing %d of %d' % (ix, len(dirs)))
    if not os.path.exists(os.path.join(d, training_inputs_fn)):
        print('Skipping %s!' % d)
        continue
    df = pd.read_csv(os.path.join(d, training_inputs_fn))
    df_subset = pd.DataFrame()
    yaws = np.zeros(len(df))
    for ix in range(len(yaws)):
        quat = [df['field.quaternion.x'][ix], df['field.quaternion.y'][ix], df['field.quaternion.z'][ix], df['field.quaternion.w'][ix]]
        yaws[ix] = euler_from_quaternion(quat)[2]

    vx_body = df.vx*np.cos(-yaws) - df.vy*np.sin(-yaws)
    vy_body = df.vx*np.sin(-yaws) + df.vy*np.cos(-yaws)

    #df_subset['vx'] = df.vx
    #df_subset['vy'] = df.vy
    df_subset['vx'] = vx_body
    df_subset['vy'] = vy_body
    df_subset['vz'] = df.vz
    df_subset['omega_z'] = df.omega_z
    df_subset.to_csv(d + '/data_out.csv', index=False)
