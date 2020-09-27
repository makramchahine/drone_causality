import os
import numpy as np
import matplotlib.pyplot as plt
import re
import PIL.Image
import random

RECORDING_DIRECTORY      = 'C:\\Users\\MIT Driverless\\Documents\\AirSim\\'
PROCESSED_DATA_DIRECTORY = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\processed-data'

TRAINING_SEQUENCE_LENGTH = 32
PLOT_STATISTICS          = False
IMAGE_SHAPE              = (256, 256, 3)

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

    # Skip folders that don't contain unprocessed recordings
    if not re.match(r'^[\-0-9]+$', runDirectory):
        print('Skipping ', runDirectory)
        continue

    imageDirectory = RECORDING_DIRECTORY + '\\' + runDirectory + '\\images'
    odometryFile   = RECORDING_DIRECTORY + '\\' + runDirectory + '\\airsim_rec.txt'

    print(f"Processing {n}/{sequenceCount} sequences")

    # Load the data, discard corrupt images
    odometry = np.array(np.genfromtxt(fname=odometryFile, dtype=imageOdometryDataType, skip_header=1))
    validImages = np.full((len(odometry),), fill_value=False)

    imageMap = dict()
    for i, record in enumerate(odometry):
        imageFile = str(record['imagefile'])
        try:
            imageMap[imageFile] = np.array(PIL.Image.open(imageDirectory + '\\' + imageFile).convert('RGB'), dtype=np.float32) / 255
            validImages[i]      = True

        except PIL.UnidentifiedImageError:
            numCorruptedSequences += 1 # skip images which are corrupted

    odometry = odometry[validImages]

    # Make the processed data directories
    try:
        os.mkdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory)
    except FileExistsError:
        continue

    # Select a sequence of TRAINING_SEQUENCE_LENGTH from the run
    runLength = odometry.shape[0]
    if runLength < TRAINING_SEQUENCE_LENGTH + 1: # +1 because image[i] is used to predict position[i+1]
        os.rmdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory)
        continue # skip runs that aren't long enough

    try:
        sequenceStart = random.randrange(runLength - TRAINING_SEQUENCE_LENGTH)
    except ValueError: 
        sequenceStart = 0 # if runLength == TRAINING_SEQUENCE_LENGTH, randrange complains

    imageSequence     = np.empty((TRAINING_SEQUENCE_LENGTH, *IMAGE_SHAPE)) 
    directionsSequence = np.empty((TRAINING_SEQUENCE_LENGTH, 3))

    for j in range(0, TRAINING_SEQUENCE_LENGTH):
        record = odometry[sequenceStart + j]
        imageFile = str(record['imagefile'])
        imageSequence[j] = imageMap[imageFile]

    np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\images.npy', imageSequence)    

    for j in range(0, TRAINING_SEQUENCE_LENGTH):
        record1 = odometry[sequenceStart + j]
        record2 = odometry[sequenceStart + j + 1] # +1 because image[i] is used to predict position[i+1]

        # Train against normalized displacement vector
        displacement = np.array([record2['x'] - record1['x'], record2['y'] - record1['y'], record2['z'] - record1['z']])
        direction = displacement / np.linalg.norm(displacement)
        directionsSequence[j] = np.array([record2['x'] - record1['x'], record2['y'] - record1['y'], record2['z'] - record1['z']])

    np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\positions.npy', directionsSequence)    

print("Proportion of sequences with corrupted images: ", numCorruptedSequences / len(sequenceLengths))

# Validate
for runDirectory in os.listdir(PROCESSED_DATA_DIRECTORY):
    files = set(os.listdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory))
    if files != {'images.npy', 'positions.npy'}:
        raise ValueError(runDirectory + ' doesn\'t have correct files.')





