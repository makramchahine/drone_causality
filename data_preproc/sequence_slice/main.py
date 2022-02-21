import cv2
import numpy as np
import os
import pandas as pd

# path = '/home/makramchahine/repos/drone_data_sequencer/data/'
# proc = '/home/makramchahine/repos/drone_data_sequencer/data_sliced/'
path = '/media/ita/BackendData/Devens-02-16-22'
proc = '/media/ita/BackendData/sliced_02-16-22'
print(proc)
os.system('mkdir ' + proc)

L = os.listdir(path)
print(L)
zoom = 3

for flight in L:
    index = 0
    seq = [[],[]]
    io = 0

    while True:
        try:
            im_num = str(index).zfill(6) # format index with zeros to the left
            img = cv2.imread(path+flight + '/' + im_num + '.png') # load frame
            height, width, channels = img.shape
            imB = cv2.resize(img,(zoom*width,zoom*height))
            cv2.imshow('img',imB)

            k = cv2.waitKey(0)
            if k == 83: # right key for next image
                index += 1
            elif k == 81: # left key for previous image
                index = max([0, index-1])
            elif k == 32: # space bar to delimit sequence
                seq[io].append(index)
                io = 1-io
                index += 1
                print(seq)
            elif k == 27:  # escape to exit
                break
            else:
                pass

        except:
            print('Flight end at frame number ' + str(index-1))
            if len(seq[0]) > len(seq[1]):
                print('Please end open sequence')
                index -= 1
            else:
                break

    ####################################################################################################################
    ### Create a folder per sequence with the corresponding images and sliced csv labels ###
    ####################################################################################################################

    n_seq = len(seq[0])
    data = pd.read_csv(path + '/' + flight + '/data_out.csv')

    for i in range(0,n_seq):
        en = seq[0][i]
        ex = seq[1][i]
        fname = flight+'_'+str(i+1)
        os.system('mkdir '+proc+fname)
        df = data[en:ex]
        df.to_csv(proc+fname+'/data_out.csv', index=False)
        # need to loop to renumber images
        for j in range(en,ex):
            os.system('cp ' + path+flight + '/' + str(j).zfill(6)+'.png ' + proc+fname + '/'+ str(j - en).zfill(6) + '.png')
