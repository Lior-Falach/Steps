#!/usr/bin/env python3
import numpy as np
#from numpy import dot, zeros, eye
#import scipy.linalg as linalg
from imuC2012 import F_func, H_func
from unitree_legged_msgs.msg import A1HighState
from unitree_legged_msgs.msg import A1True
from q_functions import q_mult, w2zq, q2R
import rospy
from std_msgs.msg import Int16MultiArray, Float32MultiArray
import time
dt = 1 / 200.0
gravity = np.array([0, 0, -9.8])

class V_ele():
    def __init__(self,k):
        self.pri=np.zeros(k)
        self.pos=np.zeros(k)
class State_ele():
    def __init__(self):
        self.r=V_ele(3)
        self.v=V_ele(3)
        self.q=V_ele(4)
        self.prf = V_ele(3)
        self.plf = V_ele(3)
        self.prr = V_ele(3)
        self.plr = V_ele(3)
        self.bf = V_ele(3)
        self.bw = V_ele(3)
class Err_State_ele():
    def __init__(self):
        self.total = np.zeros(27)
        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.zeros(3)
        self.prf = np.zeros(3)
        self.plf = np.zeros(3)
        self.prr = np.zeros(3)
        self.plr = np.zeros(3)
        self.bf = np.zeros(3)
        self.bw = np.zeros(3)
class Cov_state_ele():
    def __init__(self, P0):
        self.pri = P0
        self.pos = P0



class Est(object):

    def __init__(self, x_0, P_0 ):  # Q_0, R_0 ,f_func , h_func, F_func, H_func ):
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
        self.ros_sub_state = rospy.Subscriber("/high_state", A1HighState, self.state_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state Low res correctly initialized")
        self.ros_pub_state = rospy.Publisher("/imuC12", A1True, queue_size=1)
        rospy.loginfo("> Publisher to imuC12 correctly initialized")
        self.ros_pub_touch = rospy.Publisher("/touch", Float32MultiArray, queue_size=1)
        rospy.loginfo("> Publisher to touch correctly initialized")
        self._last_time_state_rcv = time.time()

        # Initialing  the state variables
        self.x = State_ele()

        # Initialiizing the error state variables
        self.dx=Err_State_ele()
        self.dx.total = np.zeros(27)

        # Initializing the Covariance elements
        self.P = Cov_state_ele(P_0)
        #self.P.pri = np.array(P_0)
        #self.P.pos = np.array(P_0)

        # Initializing the measurments
        self.IMU_a = np.array([0, 0, 0])
        self.IMU_g = np.array([0, 0, 0])
        self.alpha=0.5
        self.Foot_force = np.array([0, 0, 0, 0])
        self.Foot_force_bias = np.array([0, 0, 0, 0])
        self.Contact = Float32MultiArray()
        self.Contact.data=[0.0,0.0,0.0,0.0]
        self.TH = 15  # Foot force threshold
        self.Q_f = 0.1*np.eye(3)
        self.Q_bf = 0.01*np.eye(3)
        self.Q_w = 0.1*np.eye(3)
        self.Q_bw = 0.1*np.eye(3)
        self.Q_p=0.1*np.eye(3)
        self.s1 = np.array([0, 0, 0])
        self.s2 = np.array([0, 0, 0])
        self.s3 = np.array([0, 0, 0])
        self.s4 = np.array([0, 0, 0])

        #self.Q = Q_0
        self.R = 0.1*np.eye(12)

        self.F_func = F_func
        self.H_func = H_func
        self.K = np.zeros([12, 27])  # add dimentions
        # Setting up the ROS part
        self.ros_sub_state = rospy.Subscriber("/low_state_Lo_res", A1HighState, self.state_update, queue_size=1)
        rospy.loginfo("> Subscriber to low_state correctly initialized")


    def state_update(self, message):
        self.IMU_a = np.array(message.accel)
        rospy.loginfo(self.IMU_a)
        self.IMU_g = np.array(message.gyro)
        self.Foot_force_update(message.footForce)
        self.predict_state()
        self.s1 = np.array(message.rf_P)
        self.s2 = np.array(message.lf_P)
        self.s3 = np.array(message.rr_P)
        self.s4 = np.array(message.lr_P)
        self.H, self.y = self.H_func(np.array(q2R(self.x.q.pos)), self.x.r.pri, self.x.prf.pri, self.x.plf.pri, self.x.prr.pri,
                                     self.x.plr.pri, self.s1, self.s2, self.s3, self.s4)
        # The Inovation Covariance
        self.S = np.matmul(np.matmul(self.H, self.P.pri), self.H.transpose()) + self.R
        # The Kalman Gain
        self.K = np.matmul(np.matmul(self.P.pri, self.H.transpose()), np.linalg.inv(self.S))
        self.dx.total = np.matmul(self.K, self.y)
        self.dx_parts_update()
        self.P.pos = np.matmul(np.eye(27) - np.matmul(self.K, self.H), self.P.pri)
        self.post_state_update()

    def predict_state(self):
        "Performs the state prediction of the extended Kalman"
        # Get the coardinate transformation matrix
        C = np.array(q2R(self.x.q.pos))
        # State update
        self.x.r.pri = self.x.r.pos + dt * self.x.v.pos + 0.5 * dt * dt * (
                    np.matmul(C.transpose(), self.IMU_a - self.x.bf.pos) + gravity)
        self.x.v.pri = self.x.v.pos + dt * (np.matmul(C, self.IMU_a - self.x.bf.pos) + gravity)
        self.x.q.pri = q_mult(w2zq(dt * (self.IMU_g - self.x.bw.pos)), self.x.q.pos)
        self.x.prf.pri = self.x.prf.pos
        self.x.plf.pri = self.x.plf.pos
        self.x.prr.pri = self.x.prr.pos
        self.x.plr.pri = self.x.plr.pos
        # Process covariance and linearized dynamics
        self.F, self.Q = self.F_func(dt, self.IMU_a - self.x.bf.pos, C, self.IMU_g - self.x.bw.pos,self.Q_f,self.Q_bf,self.Q_w,self.Q_bw,self.Q_p,self.Contact.data)
        # prior state covariance
        self.P.pri = np.matmul(np.matmul(self.F.transpose(), self.P.pos), self.F) + self.Q

    def Foot_force_update(self, FF): # update the foot contact states

        for i in range(4):
            self.Foot_force[i] = self.alpha * self.Foot_force[i] + (1 - self.alpha) * FF[i]
            if ((self.Foot_force[i]-self.Foot_force_bias[i]) >= self.TH):
                self.Contact.data[i] = 1.0
            else:
                self.Contact.data[i] = 0.0

    def dx_parts_update(self): #Segementing the error state into components
        self.dx.r = self.dx.total[0:3]
        self.dx.v = self.dx.total[3:6]
        self.dx.q = self.dx.total[6:9]
        self.dx.prf = self.dx.total[9:12]
        self.dx.plf = self.dx.total[12:15]
        self.dx.prr = self.dx.total[15:18]
        self.dx.plr = self.dx.total[18:21]
        self.dx.bf = self.dx.total[21:24]
        self.dx.bw = self.dx.total[24:27]

    def post_state_update(self): #Posterior update once the error state has been set
        self.x.r.pos = self.x.r.pri + self.dx.r
        self.x.v.pos = self.x.v.pri + self.dx.v
        self.x.q.pos = q_mult(w2zq(self.dx.q), self.x.q.pri)
        self.x.prf.pos = self.x.prf.pri + self.dx.prf
        self.x.plf.pos = self.x.plf.pri + self.dx.plf
        self.x.prr.pos = self.x.prr.pri + self.dx.prr
        self.x.plr.pos = self.x.plr.pri + self.dx.plr
        self.x.bf.pos = self.x.bf.pri + self.dx.bf
        self.x.bw.pos = self.x.bw.pri + self.dx.bw
        rospy.loginfo(self.x.r.pos)
    def run(self):

        # --- Set the control rate
        rate = rospy.Rate(200)

        while not rospy.is_shutdown():

            self.ros_pub_touch.publish(self.Contact)

            # rospy.loginfo(self.Contact)
            #if (time.time()-self.rcv_time>self.dur_time):
                #self.cmd = [0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0]
            #self.a1.high_command(self.cmd)

            # Sleep
            rate.sleep()

# This should ve replaced with a proper initialization alg base of forward kinematics and imu calibration
P_0 = np.diag([0.1] * 27, 0)
X_0 = np.zeros(28)
X_0[9] = 1

if __name__ == "__main__":
    Cont_Est = Est(X_0,P_0)
    Cont_Est.run()
