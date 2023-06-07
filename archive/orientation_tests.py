#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tf.transformations import euler_from_quaternion

fn = '/media/aaron/BackendData/devens_2021-08-04/frames/1628106140.64/training_inputs_w_attitude.csv'
df = pd.read_csv(fn)

eulers = np.zeros((len(df), 3))
for ix in range(len(eulers)):
    quat = [df['field.quaternion.x'][ix], df['field.quaternion.y'][ix], df['field.quaternion.z'][ix], df['field.quaternion.w'][ix]]
    eulers[ix] = euler_from_quaternion(quat)

plt.plot(eulers[:,0])
plt.plot(eulers[:,1])
plt.plot(eulers[:,2])
plt.show()

yaws = eulers[:,2]

vx = df.vx
vy = df.vy

vx_new = vx*np.cos(-yaws) - vy*np.sin(-yaws)
vy_new = vx*np.sin(-yaws) + vy*np.cos(-yaws)

plt.plot(vx_new)
plt.plot(vy_new)
plt.legend(['vx', 'vy'])
plt.show()
