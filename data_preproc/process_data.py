import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import shutil
from PIL import Image
import PIL.ImageOps
import PIL

#dirs = os.listdir('raw_data')
#dirs = sorted([d for d in dirs if 'csv' not in d])
#
#if not os.path.exists('processed_data'):
#    os.mkdir('processed_data')
#
#
#bad_runs = []
#plt.ion()
#for d in dirs:
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


#bad_runs = []
bad_runs = ['1635514436.968792', '1635514761.438539', '1635516913.005976', '1635519448.436114', '1635519793.341778', '1635520665.428093', '1635521795.719742', '1635522904.005537']
good_runs = ['1635514982.189598', '1635515333.207994', '1635515509.400908', '1635515879.691413', '1635516036.155236', '1635516248.634311', '1635516532.248789', '1635516733.666543', '1635518511.425622', '1635518713.671926', '1635519064.258068', '1635519177.965528', '1635519264.042842', '1635519341.384547', '1635519662.054904', '1635520100.498251', '1635520259.923596', '1635520398.072787', '1635520539.811585', '1635521428.841970', '1635521555.476208', '1635521670.852508', '1635521931.968938', '1635522062.589399', '1635522189.141633', '1635522313.856615', '1635522417.627034', '1635522534.977889', '1635522647.587541', '1635522778.542632', '1635523029.042847', '1635523142.409256', '1635523524.677079', '1635523638.367814', '1635523762.135747', '1635523869.682094', '1635523982.234954', '1635524099.477896', '1635524230.785502', '1635524355.788831', '1635524455.868472', '1635524587.777712', '1635524711.232233', '1635524825.716155', '1635524963.700525', '1635525095.655575', '1635525213.046331', '1635525341.284985']
#good_runs = [d for d in dirs if d not in bad_runs]

print('Bad runs: ', bad_runs)
print('Good runs: ', good_runs)

for (run_ix, d) in enumerate(good_runs):
    
    print('%s (%d of %d)' % (d, run_ix, len(good_runs)))
    df = pd.read_csv(os.path.join('raw_data', '%.2f.csv' % float(d)))

    yaws = np.zeros(len(df))
    for ix in range(len(yaws)):
        quat = [df['att_x'][ix], df['att_y'][ix], df['att_z'][ix], df['att_w'][ix]]
        yaws[ix] = euler_from_quaternion(quat)[2]

    vx_body = df.vx*np.cos(-yaws) - df.vy*np.sin(-yaws)
    vy_body = df.vx*np.sin(-yaws) + df.vy*np.cos(-yaws)
    df_training = pd.DataFrame()
    df_training['vx'] = vx_body
    df_training['vy'] = vy_body
    df_training['vz'] = df.vz
    df_training['omega_z'] = df.ang_vel_z

    if not os.path.exists(os.path.join('processed_data', '%.2f' % float(d))):
        os.mkdir(os.path.join('processed_data', '%.2f' % float(d)))
    df_training.to_csv(os.path.join('processed_data', '%.2f' % float(d), 'data_out.csv'), index=False)

    img_files = sorted(os.listdir(os.path.join('raw_data', d)))

    for (ix, fn) in enumerate(img_files):
        im = Image.open(os.path.join('raw_data', d, fn))
        im_smaller = Image.fromarray(np.array(im.resize((256, 144),resample=PIL.Image.BICUBIC))[:,:,::-1])
        im_smaller.save(os.path.join('processed_data', '%.2f' % float(d), '%06d.png' % ix))


