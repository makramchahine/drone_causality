#!/usr/bin/python3

import os
import numpy as np


dirs = os.listdir('.')
for d in dirs:
    #n_frames = len(os.listdir(d)) - 1
    n_frames = len([fn for fn in os.listdir(d) if 'png' in fn])
    #with open(d + '/training_inputs.csv', 'r') as fo:
    #    lines = fo.readlines()
    labels = np.genfromtxt(d + '/data_out.csv', delimiter=',', skip_header=1)
    if np.isnan(labels).any():
        print('NaN detected! %s' % d)
        print(np.where(np.isnan(labels)))
    n_labels = len(labels)
    if n_frames == n_labels:
        print(d + ' is ok')
    else:
        print('Error for ' + d + '. Got %d frames and %d inputs' % (n_frames, n_labels))
