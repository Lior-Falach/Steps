#!/usr/bin/python
import rospy
import time
from unitree_legged_msgs.msg import A1LowState
from std_msgs.msg import Int16MultiArray
import numpy as np




# The class ServoConvert is meant to convert desired angels in real world to pwm
# commands taking into acount the srevo position direction and range

# Defining the Cont class
class Cont_est:
    def __init__(self):
        self._last_time_rcv = time.time()
        self._timeout_s = 60

        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('Contact_Est')
        # --- Create the Subscriber to joint_ang  topic
        self.ros_sub_state = rospy.Subscriber("/low_state", A1LowState, self.Contact_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state correctly initialized")
        self.ros_pub_touch = rospy.Publisher("/touch", Int16MultiArray, queue_size=1)
        rospy.loginfo("> Publisher to touch correctly initialized")
        self._last_time_state_rcv = time.time()

        self.alpha = 0.5 # Smoothing factor for the estimate
        self.F_force = np.array([0, 0, 0, 0]) # The Feet Force sensors estimation
        self._F_contact = Int16MultiArray()
        self.TH=10 #Should be determined empirically






    def Contact_update(self, message):# update the foot force estimation
        self._last_time_rcv = time.time()
        self.F_force = (1-self.alpha)*self.F_force+self.alpha*message.footForce
        self._F_contact.data = self.F_force>=self.TH
        self.ros_pub_touch.publish(self._F_contact)


    def is_controller_connected(self):
        # print time.time() - self._last_time_cmd_rcv
        return (time.time() - self._last_time_cmd_rcv < self._timeout_s)

    def run(self):

        # --- Set the control rate
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            # rospy.loginfo(self.robot_state)

            if not self.is_controller_connected:
                rospy.loginfo("Controller not connected")
                # self.set_actuators_idle()
            rate.sleep()


if __name__ == "__main__":
    Cont_Est = Cont_est()
    Cont_Est.run()

