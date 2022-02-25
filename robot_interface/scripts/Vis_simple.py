#!/usr/bin/python
import rospy
import time
from unitree_legged_msgs.msg import A1LowState

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Models import A1_FIK


# The class ServoConvert is meant to convert desired angels in real world to pwm
# commands taking into acount the srevo position direction and range

# Defining the A1_plot class
class A1_plot:
    def __init__(self):

        self.FK=A1_FIK()
        for ii in range(4):
            self.FK.Forw_kin(ii)

        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('A1_plot')

        # --- Create the Subscriber to joint_ang  topic
        self.ros_sub_state = rospy.Subscriber("/low_state", A1LowState, self.plot_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state correctly initialized")
        self._last_time_state_rcv = time.time()

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.axes.set_xlim3d(left=-0.2, right=0.2)
        self.ax.axes.set_ylim3d(bottom=-0.2, top=0.2)
        self.ax.axes.set_zlim3d(bottom=-0.2, top=0.1)
        self.ax.view_init(0, -90)

        body_x = np.array([self.FK.B_L / 2, self.FK.B_L / 2, -self.FK.B_L / 2, -self.FK.B_L / 2, self.FK.B_L / 2])
        body_y = np.array([self.FK.B_W / 2, -self.FK.B_W / 2, -self.FK.B_W / 2, self.FK.B_W / 2, self.FK.B_W / 2])
        body_z = np.array([0, 0, 0, 0, 0])
        self.LB, = self.ax.plot3D(body_x, body_y, body_z, 'gray')

        # LF
        LF_S_x = np.array([body_x[0], self.FK.S_pos[0, 0]])
        LF_S_y = np.array([body_y[0], self.FK.S_pos[0, 1]])
        LF_S_z = np.array([0, self.FK.S_pos[0, 2]])
        self.LFS, = self.ax.plot3D(LF_S_x, LF_S_y, LF_S_z, 'green')
        LF_H_x = np.array([self.FK.S_pos[0, 0], self.FK.K_pos[0, 0]])
        LF_H_y = np.array([self.FK.S_pos[0, 1], self.FK.K_pos[0, 1]])
        LF_H_z = np.array([self.FK.S_pos[0, 2], self.FK.K_pos[0, 2]])
        self.LFH, = self.ax.plot3D(LF_H_x, LF_H_y, LF_H_z, 'red')
        LF_F_x = np.array([self.FK.K_pos[0, 0], self.FK.F_pos[0, 0]])
        LF_F_y = np.array([self.FK.K_pos[0, 1], self.FK.F_pos[0, 1]])
        LF_F_z = np.array([self.FK.K_pos[0, 2], self.FK.F_pos[0, 2]])
        self.LFF, = self.ax.plot3D(LF_F_x, LF_F_y, LF_F_z, 'blue')

        # RF
        RF_S_x = np.array([body_x[1], self.FK.S_pos[1, 0]])
        RF_S_y = np.array([body_y[1], self.FK.S_pos[1, 1]])
        RF_S_z = np.array([0, self.FK.S_pos[1, 2]])
        self.RFS, = self.ax.plot3D(RF_S_x, RF_S_y, RF_S_z, 'green')
        RF_H_x = np.array([self.FK.S_pos[1, 0], self.FK.K_pos[1, 0]])
        RF_H_y = np.array([self.FK.S_pos[1, 1], self.FK.K_pos[1, 1]])
        RF_H_z = np.array([self.FK.S_pos[1, 2], self.FK.K_pos[1, 2]])
        self.RFH, = self.ax.plot3D(RF_H_x, RF_H_y, RF_H_z, 'red')
        RF_F_x = np.array([self.FK.K_pos[1, 0], self.FK.F_pos[1, 0]])
        RF_F_y = np.array([self.FK.K_pos[1, 1], self.FK.F_pos[1, 1]])
        RF_F_z = np.array([self.FK.K_pos[1, 2], self.FK.F_pos[1, 2]])
        self.RFF, = self.ax.plot3D(RF_F_x, RF_F_y, RF_F_z, 'blue')

        # LR
        LR_S_x = np.array([body_x[3], self.FK.S_pos[2, 0]])
        LR_S_y = np.array([body_y[3], self.FK.S_pos[2, 1]])
        LR_S_z = np.array([0, self.FK.S_pos[2, 2]])
        self.LRS, = self.ax.plot3D(LR_S_x, LR_S_y, LR_S_z, 'green')
        LR_H_x = np.array([self.FK.S_pos[2, 0], self.FK.K_pos[2, 0]])
        LR_H_y = np.array([self.FK.S_pos[2, 1], self.FK.K_pos[2, 1]])
        LR_H_z = np.array([self.FK.S_pos[2, 2], self.FK.K_pos[2, 2]])
        self.LRH, = self.ax.plot3D(LR_H_x, LR_H_y, LR_H_z, 'red')
        LR_F_x = np.array([self.FK.K_pos[2, 0], self.FK.F_pos[2, 0]])
        LR_F_y = np.array([self.FK.K_pos[2, 1], self.FK.F_pos[2, 1]])
        LR_F_z = np.array([self.FK.K_pos[2, 2], self.FK.F_pos[2, 2]])
        self.LRF, = self.ax.plot3D(LR_F_x, LR_F_y, LR_F_z, 'blue')
        # RR
        RR_S_x = np.array([body_x[2], self.FK.S_pos[3, 0]])
        RR_S_y = np.array([body_y[2], self.FK.S_pos[3, 1]])
        RR_S_z = np.array([0, self.FK.S_pos[3, 2]])
        self.RRS, = self.ax.plot3D(RR_S_x, RR_S_y, RR_S_z, 'green')
        RR_H_x = np.array([self.FK.S_pos[3, 0], self.FK.K_pos[3, 0]])
        RR_H_y = np.array([self.FK.S_pos[3, 1], self.FK.K_pos[3, 1]])
        RR_H_z = np.array([self.FK.S_pos[3, 2], self.FK.K_pos[3, 2]])
        self.RRH, = self.ax.plot3D(RR_H_x, RR_H_y, RR_H_z, 'red')
        RR_F_x = np.array([self.FK.K_pos[3, 0], self.FK.F_pos[3, 0]])
        RR_F_y = np.array([self.FK.K_pos[3, 1], self.FK.F_pos[3, 1]])
        RR_F_z = np.array([self.FK.K_pos[3, 2], self.FK.F_pos[3, 2]])
        self.RRF, = self.ax.plot3D(RR_F_x, RR_F_y, RR_F_z, 'blue')
        plt.show()



    def plot_update(self, message):
        self.FK.Joint_Ang[0, 0] = message.q[3]
        self.FK.Joint_Ang[0, 1] = message.q[4]
        self.FK.Joint_Ang[0, 2] = message.q[5]

        self.FK.Joint_Ang[1, 0] = message.q[0]
        self.FK.Joint_Ang[1, 1] = message.q[1]
        self.FK.Joint_Ang[1, 2] = message.q[2]

        self.FK.Joint_Ang[2, 0] = message.q[9]
        self.FK.Joint_Ang[2, 1] = message.q[10]
        self.FK.Joint_Ang[2, 2] = message.q[11]

        self.FK.Joint_Ang[3, 0] = message.q[6]
        self.FK.Joint_Ang[3, 1] = message.q[7]
        self.FK.Joint_Ang[3, 2] = message.q[8]

        for ii in range(4):
            self.FK.Forw_kin(ii)
        self.plot_spot_update()




    def plot_spot_update(self):
        # rospy.loginfo(self.F_pos[3,:])
        body_x = np.array([self.B_L / 2, self.B_L / 2, -self.B_L / 2, -self.B_L / 2, self.B_L / 2])
        body_y = np.array([self.B_W / 2, -self.B_W / 2, -self.B_W / 2, self.B_W / 2, self.B_W / 2])
        body_z = np.array([0, 0, 0, 0, 0])
        self.LB.set_xdata(body_x)
        self.LB.set_ydata(body_y)
        self.LB.set_3d_properties(body_z)

        # LF
        LF_S_x = np.array([body_x[0], self.FK.S_pos[0, 0]])
        LF_S_y = np.array([body_y[0], self.FK.S_pos[0, 1]])
        LF_S_z = np.array([0, self.FK.S_pos[0, 2]])
        self.LFS.set_xdata(LF_S_x)
        self.LFS.set_ydata(LF_S_y)
        self.LFS.set_3d_properties(LF_S_z)
        LF_H_x = np.array([self.FK.S_pos[0, 0], self.FK.K_pos[0, 0]])
        LF_H_y = np.array([self.FK.S_pos[0, 1], self.FK.K_pos[0, 1]])
        LF_H_z = np.array([self.FK.S_pos[0, 2], self.FK.K_pos[0, 2]])
        self.LFH.set_xdata(LF_H_x)
        self.LFH.set_ydata(LF_H_y)
        self.LFH.set_3d_properties(LF_H_z)
        LF_F_x = np.array([self.FK.K_pos[0, 0], self.FK.F_pos[0, 0]])
        LF_F_y = np.array([self.FK.K_pos[0, 1], self.FK.F_pos[0, 1]])
        LF_F_z = np.array([self.FK.K_pos[0, 2], self.FK.F_pos[0, 2]])
        self.LFF.set_xdata(LF_F_x)
        self.LFF.set_ydata(LF_F_y)
        self.LFF.set_3d_properties(LF_F_z)

        # RF
        RF_S_x = np.array([body_x[1], self.FK.S_pos[1, 0]])
        RF_S_y = np.array([body_y[1], self.FK.S_pos[1, 1]])
        RF_S_z = np.array([0, self.FK.S_pos[1, 2]])
        self.RFS.set_xdata(RF_S_x)
        self.RFS.set_ydata(RF_S_y)
        self.RFS.set_3d_properties(RF_S_z)
        RF_H_x = np.array([self.FK.S_pos[1, 0], self.FK.K_pos[1, 0]])
        RF_H_y = np.array([self.FK.S_pos[1, 1], self.FK.K_pos[1, 1]])
        RF_H_z = np.array([self.FK.S_pos[1, 2], self.FK.K_pos[1, 2]])
        self.RFH.set_xdata(RF_H_x)
        self.RFH.set_ydata(RF_H_y)
        self.RFH.set_3d_properties(RF_H_z)
        RF_F_x = np.array([self.FK.K_pos[1, 0], self.FK.F_pos[1, 0]])
        RF_F_y = np.array([self.FK.K_pos[1, 1], self.FK.F_pos[1, 1]])
        RF_F_z = np.array([self.FK.K_pos[1, 2], self.FK.F_pos[1, 2]])
        self.RFF.set_xdata(RF_F_x)
        self.RFF.set_ydata(RF_F_y)
        self.RFF.set_3d_properties(RF_F_z)

        # LR
        LR_S_x = np.array([body_x[3], self.FK.S_pos[2, 0]])
        LR_S_y = np.array([body_y[3], self.FK.S_pos[2, 1]])
        LR_S_z = np.array([0, self.FK.S_pos[2, 2]])
        self.LRS.set_xdata(LR_S_x)
        self.LRS.set_ydata(LR_S_y)
        self.LRS.set_3d_properties(LR_S_z)
        LR_H_x = np.array([self.FK.S_pos[2, 0], self.FK.K_pos[2, 0]])
        LR_H_y = np.array([self.FK.S_pos[2, 1], self.FK.K_pos[2, 1]])
        LR_H_z = np.array([self.FK.S_pos[2, 2], self.FK.K_pos[2, 2]])
        self.LRH.set_xdata(LR_H_x)
        self.LRH.set_ydata(LR_H_y)
        self.LRH.set_3d_properties(LR_H_z)
        LR_F_x = np.array([self.FK.K_pos[2, 0], self.FK.F_pos[2, 0]])
        LR_F_y = np.array([self.FK.K_pos[2, 1], self.FK.F_pos[2, 1]])
        LR_F_z = np.array([self.FK.K_pos[2, 2], self.FK.F_pos[2, 2]])
        self.LRF.set_xdata(LR_F_x)
        self.LRF.set_ydata(LR_F_y)
        self.LRF.set_3d_properties(LR_F_z)
        # RR
        RR_S_x = np.array([body_x[2], self.FK.S_pos[3, 0]])
        RR_S_y = np.array([body_y[2], self.FK.S_pos[3, 1]])
        RR_S_z = np.array([0, self.FK.S_pos[3, 2]])
        self.RRS.set_xdata(RR_S_x)
        self.RRS.set_ydata(RR_S_y)
        self.RRS.set_3d_properties(RR_S_z)

        RR_H_x = np.array([self.FK.S_pos[3, 0], self.FK.K_pos[3, 0]])
        RR_H_y = np.array([self.FK.S_pos[3, 1], self.FK.K_pos[3, 1]])
        RR_H_z = np.array([self.FK.S_pos[3, 2], self.FK.K_pos[3, 2]])
        self.RRH.set_xdata(RR_H_x)
        self.RRH.set_ydata(RR_H_y)
        self.RRH.set_3d_properties(RR_H_z)

        RR_F_x = np.array([self.FK.K_pos[3, 0], self.FK.F_pos[3, 0]])
        RR_F_y = np.array([self.FK.K_pos[3, 1], self.FK.F_pos[3, 1]])
        RR_F_z = np.array([self.FK.K_pos[3, 2], self.FK.F_pos[3, 2]])
        self.RRF.set_xdata(RR_F_x)
        self.RRF.set_ydata(RR_F_y)
        self.RRF.set_3d_properties(RR_F_z)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def is_controller_connected(self):
        # print time.time() - self._last_time_cmd_rcv
        return (time.time() - self._last_time_cmd_rcv < self._timeout_s)

    def run(self):

        # --- Set the control rate
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # rospy.loginfo(self.robot_state)

            if not self.is_controller_connected:
                rospy.loginfo("Controller not connected")
                # self.set_actuators_idle()
            rate.sleep()


if __name__ == "__main__":
    A1_vis = A1_plot()
    A1_vis.run()

