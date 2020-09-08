import os
import numpy as np
import matplotlib.pyplot as plt
import re
import PIL.Image
import random

RECORDING_DIRECTORY      = 'C:\\Users\\MIT Driverless\\Documents\\AirSim\\'
PROCESSED_DATA_DIRECTORY = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\processed-data'

TRAINING_SEQUENCE_LENGTH = 101      # Remember to add +1 because image[i] is used to predict position[i+1]
PLOT_STATISTICS          = False


imageOdometryDataType = [('timestamp', np.uint64), ('x', np.float32), ('y', np.float32), ('z', np.float32), ('qw', np.float32), ('qx', np.float32), ('qy', np.float32), ('qz', np.float32), ('imagefile', 'U32')]

# iterate over contents of recording directory
sequenceLengths = []
for runDirectory in os.listdir(RECORDING_DIRECTORY):

    # Skip folders that don't contain unprocessed recordings
    if not re.match(r'^[\-0-9]+$', runDirectory):
        print('Skipping ', runDirectory)
        continue

    imageDirectory = RECORDING_DIRECTORY + '\\' + runDirectory + '\\images'
    n = len([image for image in os.listdir(imageDirectory)])
    sequenceLengths.append(n)

sequenceCount = len(sequenceLengths)

plt.hist(sequenceLengths, bins=50)
if PLOT_STATISTICS:
    plt.show()

numCorruptedSequences = 0

# Clean each run and save it to the processed data directory
for n, runDirectory in enumerate(os.listdir(RECORDING_DIRECTORY)):
    imageDirectory = RECORDING_DIRECTORY + '\\' + runDirectory + '\\images'
    odometryFile   = RECORDING_DIRECTORY + '\\' + runDirectory + '\\airsim_rec.txt'

    # Load the data, discard corrupt images
    odometry = np.array(np.genfromtxt(fname=odometryFile, dtype=imageOdometryDataType, skip_header=1))
    validImages = np.full((len(odometry),), fill_value=False)

    imageMap = dict()
    for i, record in enumerate(odometry):
        imageFile = str(record['imagefile'])
        try:
            imageMap[imageFile] = np.array(PIL.Image.open(imageDirectory + '\\' + imageFile).convert('RGB'))
            validImages[i]      = True

        except PIL.UnidentifiedImageError:
            numCorruptedSequences += 1 # skip images which are corrupted

    odometry = odometry[validImages]

    # Select a sequence of TRAINING_SEQUENCE_LENGTH from the run
    runLength = odometry.shape[0]
    if runLength < TRAINING_SEQUENCE_LENGTH:
        continue # skip runs that aren't long enough

    sequenceStart = random.randrange(runLength - TRAINING_SEQUENCE_LENGTH)

    # Make the processed data directories
    os.mkdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory)
    os.mkdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\images')

    np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\airsim_rec.npy', odometry)

    for j in range(sequenceStart, runLength):
        record = odometry[j]
        imageFile = str(record['imagefile'])
        np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\images\\' + imageFile, imageMap[imageFile])    

    print(f"Finished processing {n}/{sequenceCount} sequences")

print("Proportion of sequences with corrupted images: ", numCorruptedSequences / len(sequenceLengths))




