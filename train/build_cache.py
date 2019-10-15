import argparse
import cv2
from cv_bridge import CvBridge
from enum import Enum
import h5py
import rosbag

parser = argparse.ArgumentParser()
parser.add_argument("--bag", help="path to the bag to train with")
args = parser.parse_args()


class Sensor(Enum):
    IMAGE = 1
    ODOM = 2


topics = {
    Sensor.IMAGE: '/bebop/image_raw',
    Sensor.ODOM: '/bebop/odom',
}

bridge = CvBridge()
bag = rosbag.Bag(args.bag)

for topic, msg, t in bag.read_messages(topics=topics.values()):
    if topic == topics[Sensor.IMAGE]:
        cv_image = bridge.imgmsg_to_cv2(msg)
        cv2.imshow('hi', cv_image)
        cv2.waitKey(10)
        print(t)
    elif topic == topics[Sensor.ODOM]:
        pass
    print (t)

bag.close()
