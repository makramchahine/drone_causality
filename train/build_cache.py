import argparse
from collections import OrderedDict as odict
import numpy as np
from cv_bridge import CvBridge
from enum import Enum
import h5py
import matplotlib.pyplot as plt
import rosbag
import scipy.interpolate

parser = argparse.ArgumentParser()
parser.add_argument("--bag", help="path to the bag to train with")
parser.add_argument("--cache", default="cache.h5", help="path to save the cache")
args = parser.parse_args()


class Sensor(Enum):
    IMAGE = 1
    ODOM = 2


topics = {
    Sensor.IMAGE: '/bebop/image_raw',
    Sensor.ODOM: '/bebop/odom',
}

bag_values = {sensor: odict() for sensor in topics.keys()}

bridge = CvBridge()
bag = rosbag.Bag(args.bag)

for topic, msg, t in bag.read_messages(topics=topics.values()):
    if topic == topics[Sensor.IMAGE]:
        img = bridge.imgmsg_to_cv2(msg)
        bag_values[Sensor.IMAGE][t.to_time()] = img

    elif topic == topics[Sensor.ODOM]:
        data = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z] #save linear velocity
        bag_values[Sensor.ODOM][t.to_time()] = data

bag.close()


f_odom = scipy.interpolate.interp1d(bag_values[Sensor.ODOM].keys(), bag_values[Sensor.ODOM].values(), axis=0, fill_value='extrapolate')
def f_image(key):
    D = bag_values[Sensor.IMAGE]
    return D[min(D.keys(), key=lambda k: abs(k-key))]

sample_times = bag_values[Sensor.IMAGE].keys()
images = [bag_values[Sensor.IMAGE][t] for t in sample_times] # use f_image if not exact t
images = np.array(images) #convert to numpy array
odom = f_odom(sample_times)

plt.plot(sample_times, odom)
plt.title('odometry readings')
plt.show()

f = h5py.File(args.cache, 'w')
f.create_dataset("images", data=images)
f.create_dataset("odom", data=odom)
f.close()

