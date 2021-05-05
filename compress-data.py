import os
import shutil
import numpy as np

DATASET_PATH = 'C:\\Users\\MIT Driverless\\Documents\\AirSim\\following-neighborhood-parsed'
TRUNCATED_PATH = DATASET_PATH + '-truncated'

if not os.path.exists(TRUNCATED_PATH):
    os.makedirs(TRUNCATED_PATH)

for dataset in os.listdir(DATASET_PATH):
    print(dataset)
    data_files = set(os.listdir(DATASET_PATH + '\\' + dataset))
    for f in data_files:
        data = np.load(DATASET_PATH + '\\' + dataset + '\\' + f)
        uint8_data = (255*data).astype(np.uint8) 
        if not os.path.exists(TRUNCATED_PATH + '\\' + dataset):
            os.makedirs(TRUNCATED_PATH + '\\' + dataset)
        np.save(TRUNCATED_PATH + '\\' + dataset + '\\' + f, uint8_data)