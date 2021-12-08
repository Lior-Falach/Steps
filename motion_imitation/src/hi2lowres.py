#!/usr/bin/env python3
import rospy
from unitree_legged_msgs.msg import A1LowState
from unitree_legged_msgs.msg import A1True
import time
import numpy as np
class RobCom:
    def __init__(self):
        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('hi2low')
        # --- Create the angel array publisher
        self.ros_pub_lowstate = rospy.Publisher("/low_state_Lo_res", A1LowState, queue_size=1)  # topic, msg, queue=1=>publish every time
        self.ros_pub_GT = rospy.Publisher("/A1_GT_Lo_res", A1True,queue_size=1)  # topic, msg, queue=1=>publish every time
        rospy.loginfo("> publishers corrrectly initialized")
        self.ros_sub_lowstate = rospy.Subscriber("/low_state_Hi_res", A1LowState, self.get_low_state)
        self.ros_sub_GT = rospy.Subscriber("/A1_GT_Hi_res", A1True, self.get_GT)
        rospy.loginfo("> Subscribers corrrectly initialized")
        q=[0]
        self.gotFirst=0
        self.m=[]# q*12,q*12,q*12,q*12,q*4,q*4,q*3,q*3
    def get_low_state(self,message):
        self.time_r=time.time()
        self.m=message
        self.gotFirst=1
        #self.ros_pub_lowstate.publish(message)
    def get_GT(self,message):
        self.gt=message
        #self.ros_pub_GT.publish(message)


    def run(self):
        # --- Set the control rate
        rate=rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.gotFirst:
                self.ros_pub_lowstate.publish(self.m)
                self.ros_pub_GT.publish(self.gt)
            rate.sleep()



if __name__ == "__main__":
    robcom     =  RobCom ()
    robcom.run()