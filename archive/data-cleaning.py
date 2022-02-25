import os
import numpy as np
import matplotlib.pyplot as plt
import re
import PIL.Image
import random
from numpy.lib.function_base import disp
import rowan
from scipy.spatial.transform import Rotation as R


RECORDING_DIRECTORY      = 'C:\\Users\\MIT Driverless\\Documents\\AirSim\\demo-redwood-recent'
PROCESSED_DATA_DIRECTORY = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\training-data'

TRAINING_SEQUENCE_LENGTH = 32
PLOT_STATISTICS          = False
IMAGE_SHAPE              = (256, 256, 3)

imageOdometryDataType = [
    ('timestamp', np.uint64), 
    ('x', np.float32), 
    ('y', np.float32), 
    ('z', np.float32), 
    ('qw', np.float32), 
    ('qx', np.float32),
    ('qy', np.float32), 
    ('qz', np.float32), 
    ('imagefile', 'U32')
]

droneStateVector = [
    ('x', np.float32), 
    ('y', np.float32), 
    ('z', np.float32), 
    ('qw', np.float32), 
    ('qx', np.float32),
    ('qy', np.float32), 
    ('qz', np.float32), 
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('wx', np.float32),
    ('wy', np.float32),
    ('wz', np.float32),
]

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
    try:
        validImages = np.full((len(odometry),), fill_value=False)
    except Exception as e:
        print("Bad Run: ", runDirectory)
        #os.remove(RECORDING_DIRECTORY + '\\' + runDirectory)
        continue

    imageMap = dict()
    for i, record in enumerate(odometry):
        imageFile = str(record['imagefile'])
        try:
            imageMap[imageFile] = np.array(PIL.Image.open(imageDirectory + '\\' + imageFile).convert('RGB'), dtype=np.float32) / 255
            validImages[i]      = True

        except PIL.UnidentifiedImageError:
            numCorruptedSequences += 1 # skip images which are corrupted
        
        except FileNotFoundError:
            print('Error image: ' + imageDirectory + '\\' + imageFile + ' doesn\'t exist.')
            continue

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
        print("Skipping short sequence")
        continue # skip runs that aren't long enough

    try:
        sequenceStart = random.randrange(runLength - TRAINING_SEQUENCE_LENGTH)
    except ValueError: 
        sequenceStart = 0 # if runLength == TRAINING_SEQUENCE_LENGTH, randrange complains

    imageSequence = np.empty((TRAINING_SEQUENCE_LENGTH, *IMAGE_SHAPE)) 
    stateSequence = np.empty((TRAINING_SEQUENCE_LENGTH,), dtype=droneStateVector)

    for j in range(0, TRAINING_SEQUENCE_LENGTH):
        record = odometry[sequenceStart + j]
        imageFile = str(record['imagefile'])
        imageSequence[j] = imageMap[imageFile]

    np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\images.npy', imageSequence)    
    print("Saved images to: ", PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\images.npy')

    record0 = odometry[sequenceStart]
    origin  = np.array([record0['x'], record['y'], record['z']])
    originOrientation = R.from_quat(rowan.normalize([record0['qw'], record0['qx'], record0['qy'], record0['qz']]))

    for j in range(0, TRAINING_SEQUENCE_LENGTH):
        record1 = odometry[sequenceStart + j]
        record2 = odometry[sequenceStart + j + 1] # +1 because image[i] is used to predict position[i+1]

        # Rotate into camera frame
        orientation = R.from_quat(rowan.normalize([record1['qw'], record1['qx'], record1['qy'], record1['qz']]))
        orientation = orientation

        # displacement w.r.t. first frame
        displacement = np.array([record1['x'], record1['y'], record1['z']]) - origin
        displacement = orientation.inv().apply(displacement)

        # finite difference velocity
        velocity = np.array([record2['x'] - record1['x'], record2['y'] - record1['y'], record2['z'] - record1['z']])
        velocity = orientation.inv().apply(velocity)

        # finite difference angular velocity
        q1 = rowan.normalize(orientation.as_quat())
        q2 = rowan.normalize([record2['qw'], record2['qx'], record2['qy'], record2['qz']])

        angular_velocity = 2*rowan.log(rowan.multiply(q2, rowan.conjugate(q1)))[1:]
        angular_velocity = orientation.inv().apply(angular_velocity)

        # orientation w.r.t. first frame
        relativeOrientation = (orientation * originOrientation.inv()).as_quat()

        state = np.array((*displacement, *relativeOrientation, *velocity, *angular_velocity), dtype=droneStateVector)
        stateSequence[j] = state

    np.save(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\vectors.npy', stateSequence)    
    print("Saved vectors to: " + PROCESSED_DATA_DIRECTORY + '\\' + runDirectory + '\\vectors.npy')

print("Proportion of sequences with corrupted images: ", numCorruptedSequences / len(sequenceLengths))

# Validate
for runDirectory in os.listdir(PROCESSED_DATA_DIRECTORY):
    files = set(os.listdir(PROCESSED_DATA_DIRECTORY + '\\' + runDirectory))
    if files != {'images.npy', 'positions.npy'}:
        raise ValueError(runDirectory + ' doesn\'t have correct files.')





