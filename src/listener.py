#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import time
import cv2


bridge = CvBridge()
def callback(img_msg):
    image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

    cv2.imshow('test image', image)
    cv2.waitKey(1)

    rospy.loginfo("callback: "+str(time.time()))
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/bebop/image_raw", Image, callback)
    # rospy.Subscriber("/bebop/image_raw", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
