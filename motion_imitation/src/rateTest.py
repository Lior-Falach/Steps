#!/usr/bin/env python3
import rospy

if __name__ == '__main__':
    rospy.init_node("rospy_rate_test")
    rate = rospy.Rate(5)  # ROS Rate at 5Hz

    while not rospy.is_shutdown():
        rospy.loginfo("Hello")
        rate.sleep()