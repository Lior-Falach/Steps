#!/usr/bin/python
import rospy
import time
from unitree_legged_msgs.msg import A1LowState
from unitree_legged_msgs.msg import R_State_imuc
import numpy as np
import filterpy


# The class ServoConvert is meant to convert desired angels in real world to pwm
# commands taking into acount the srevo position direction and range

# Defining the Cont class
class EstIMUC:
    def __init__(self):
        Self.EKF=filterpy.kalman.ExtendedKalmanFilter(dim_x=28, dim_z=, dim_u=0)
        self._last_time_rcv = time.time()
        self._timeout_s = 60

        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('Est_IMUcentric')

        # --- Create the Subscriber to joint_ang  topic
        self.ros_sub_state = rospy.Subscriber("/low_state", A1LowState, self.Contact_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state correctly initialized")
        self.ros_pub_touch = rospy.Publisher("/pos_imuc", R_State_imuc, queue_size=1)
        rospy.loginfo("> Publisher to pos_imuc correctly initialized")
        self._last_time_state_rcv = time.time()
        #Setting up the Variables for the EKF
        self.x = np.zeros((28, 1)) # The State
        self.x_ = np.zeros((28, 1))  # The State Prior
        self.P = np.zeros((28, 28))  # The State covariance
        self.P_ = np.zeros((28, 28))  # The State covariance Prior
        self.K = np.zeros((12, 28))  # Kalman Gain








    def Contact_update(self, message: A1LowState):

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

