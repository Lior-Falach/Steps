# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments


"""Interface class for general purpose non linear Kalman Filter Estimation
"""

from __future__ import (absolute_import, division, unicode_literals)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from imuC2012 import f_func, h_func, F_func, H_func, Q_0, R_0
from unitree_legged_msgs.msg import A1HighState
from unitree_legged_msgs.msg import A1True
from q_functions import q_mult, w2zq, q2R

dt = 1 / 200.0
gravity = np.array([0, 0, -9.8])


class Est(object):

    def __init__(self, x_0, P_0, ):  # Q_0, R_0 ,f_func , h_func, F_func, H_func ):
        # "x_0 and P_0 are the initial state and state Covariance matrices numpy.array(dim_x,1) numpy.array(dim_x, dim_x)"
        # "Q_0 and R_0 are rge initial Process and Measurment Covariance Noise numpy.array(dim_x, dim_x) numpy.array(dim_z, dim_z)"
        # #The following four arguments should reffer to function
        # "f is a handle to the system dynamics x_k+1=f(x_k,u_k)" #x=f_func(x,u)
        # "h is the measurment function z_k=h(x_k,u_k)" #z=h_func(x,u)
        # "F is the gradient of f w.r.t. x" #F=F_func(x)
        # "H is the gradiant of h w.r.t. x" #H=H_func(x)

        # Initialization of the ROS elements
        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('imuCentrocLocalization_Est')
        # --- Create the Subscriber to the low state lo res  topic
        self.ros_sub_state = rospy.Subscriber("/low_state_Lo_res", A1HighState, self.State_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state Low res correctly initialized")
        self.ros_pub_state = rospy.Publisher("/imuC12", A1True, queue_size=1)
        rospy.loginfo("> Publisher to imuC12 correctly initialized")
        self._last_time_state_rcv = time.time()

        # Initialing  the state variables
        self.x.r.pri = np.array(x_0[0:3])
        self.x.r.pos = np.array(x_0[0:3])

        self.x.v.pri = np.array(x_0[3:6])
        self.x.v.pos = np.array(x_0[3:6])

        self.x.q.pri = np.array(x_0[6:10])
        self.x.q.pos = np.array(x_0[6:10])

        self.x.prf.pri = np.array(x_0[10:13])
        self.x.prf.pos = np.array(x_0[10:13])

        self.x.plf.pri = np.array(x_0[13:16])
        self.x.plf.pos = np.array(x_0[13:16])

        self.x.prr.pri = np.array(x_0[16:19])
        self.x.prr.pos = np.array(x_0[16:19])

        self.x.plr.pri = np.array(x_0[19:22])
        self.x.plr.pos = np.array(x_0[19:22])

        self.x.bf.pri = np.array(x_0[22:25])
        self.x.bf.pos = np.array(x_0[22:25])

        self.x.bw.pri = np.array(x_0[25:28])
        self.x.bw.pos = np.array(x_0[25:28])

        # Initialiizing the error state variables
        self.dx.total = np.zeros(27)
        self.dx_parts_update()
        # self.dx.r = np.array([0, 0, 0])
        # self.dx.v = np.array([0, 0, 0])
        # self.dx.q = np.array([0, 0, 0])
        # self.dx.prf = np.array([0, 0, 0])
        # self.dx.plf = np.array([0, 0, 0])
        # self.dx.prr = np.array([0, 0, 0])
        # self.dx.plr = np.array([0, 0, 0])
        # self.dx.bf = np.array([0, 0, 0])
        # self.dx.bw = np.array([0, 0, 0])

        # Initializing the Covariance elements

        self.P.pri = np.array(P_0)
        self.P.pos = np.array(P_0)

        # Initializing the measurments
        self.IMU_a = np.array([0, 0, 0])
        self.IMU_g = np.array([0, 0, 0])
        self.Foot_force = np.array([0, 0, 0, 0])
        self.Foot_force_bias = np.array([0, 0, 0, 0])
        self.Contact = np.array([0, 0, 0, 0])
        self.TH = 15  # Foot force threshold
        self.Q_f = 0.1*np.eye(3)
        self.Q_bf = 0.01*np.eye(3)
        self.Q_w = 0.1*np.eye(3)
        self.Q_bw = 0.1*np.eye(3)
        self.Q_p_low=0.1*np.eye(3)
        self.Q_p_high=100*np.eye(3)
        self.Q = Q_0
        self.R = R_0
        self.f_func = f_func
        self.h_func = h_func
        self.F_func = F_func
        self.H_func = H_func
        self.K = []  # "Initialize the Kalman Gain"
        # Setting up the ROS part
        # --- Create the Subscriber to joint_ang  topic
        self.ros_sub_state = rospy.Subscriber("/low_state_Lo_res", A1HighState, self.state_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state correctly initialized")
        self.ros_pub_touch = rospy.Publisher("/touch", Int16MultiArray, queue_size=1)
        rospy.loginfo("> Publisher to touch correctly initialized")
        self._last_time_state_rcv = time.time()

    def state_update(self, message):
        self.IMU_a = message.accel
        self.IMU_g = message.gyro
        self.predict_state()
        self.Foot_force_update(message.footForce)
        self.H, self.y = self.H_func(self.C, self.x.r.pri, self.x.prf.pri, self.x.plf.pri, self.x.prr.pri,
                                     self.x.prl.pri)
        # The Inovation Covariance
        self.S = np.matmul(np.matmul(self.H, self.P.pri), self.H.transpose()) + self.R
        # The Kalman Gain
        self.K = np.matmul(np.matmul(self.P.pri, self.H.transpose()), np.linalg.inv(self.S))
        self.dx.total = np.matmul(self.K, self.y)
        self.dx_parts_update()
        self.P.pos = np.matmul(np.eye() - np.matmul(self.K, self.H), self.P.pri)
        self.post_state_update()

    def predict_state(self):
        "Performs the state prediction of the extended Kalman"
        # Get the coardinate transformation matrix
        C = np.array(q2R(self.x.q.pos))
        # State update
        self.x.r.pri = self.x.r.pos + dt * self.x.v.pos + 0.5 * dt * dt * (
                    np.matmul(C.transpose(), self.IMU_a - self.x.bf.pos) + gravity)
        self.x.v.pri = self.v.pos + dt * (np.matmul(C, self.IMU_a - self.x.bf.pos) + gravity)
        self.x.q.pri = q_mult(w2zq(dt * (self.IMU_g - self.x.bw.pos)), self.x.q.pos)
        self.x.prf.pri = self.x.prf.pos
        self.x.plf.pri = self.x.plf.pos
        self.x.prr.pri = self.x.prr.pos
        self.x.plr.pri = self.x.plr.pos
        # Process covariance and linearized dynamics
        self.F, self.Q = self.F_func(dt, self.IMU_a - self.x.bf.pos, C, self.IMU_g - self.x.bw.pos)
        # prior state covariance
        self.P.pri = np.matmul(np.matmul(F.transpose(), self.P.pos), F) + self.Q

    def Foot_force_update(self, FF):
        self.Foot_force = self.alpha * self.Foot_force + (1 - self.alpha) * FF
        self.Contact = (self.Foot_force-self.Foot_force_bias) >= self.TH

    def dx_parts_update(self):
        self.dx.r = self.dx.total[0:3]
        self.dx.v = self.dx.total[3:6]
        self.dx.q = self.dx.total[6:9]
        self.dx.prf = self.dx.total[9:12]
        self.dx.plf = self.dx.total[12:15]
        self.dx.prr = self.dx.total[15:18]
        self.dx.plr = self.dx.total[18:21]
        self.dx.bf = self.dx.total[21:24]
        self.dx.bw = self.dx.total[24:27]

    def post_state_update(self):
        self.x.r.pos = self.x.r.pri + self.dx.r
        self.x.v.pos = self.v.r.pri + self.dx.v
        self.x.q.pos = q_mult(w2zq(self.dx.q), self.x.q.pri)
        self.x.prf.pos = self.x.prf.pri + self.dx.prf
        self.x.plf.pos = self.x.plf.pri + self.dx.plf
        self.x.prr.pos = self.x.prr.pri + self.dx.prr
        self.x.plr.pos = self.x.plr.pri + self.dx.plr
        self.x.bf.pos = self.x.bf.pri + self.dx.bf
        self.x.bw.pos = self.x.bw.pri + self.dx.bw


# This should ve replaced with a proper initialization alg base of forward kinematics and imu calibration
P_0 = np.diag([0.1] * 28, 0)
X_0 = np.diag([0.1] * 28, 0)
X_0[9] = 1

if __name__ == "__main__":
    Cont_Est = Cont_est()
    Cont_Est.run()
