"This file defines the relevent geometric models Forward & inverse Kinematics and Dynamics "
import numpy as np
from math import pi, cos, sin, atan2, sqrt

# Could we define the parameters (Shin_L,...,Body_L) to apply to all models?

class A1_FK_IK:

    def __init__(self):
        #Define the parameters
        self.S_L = Shin_L "Foot to Knee"
        self.T_L = Thigh_L "Knee to Hip"
        self.H_L = Hip_L "Hip to sholder"
        self.B_W = Body_W "Main body width"
        self.B_L = Body_L "Main body length"
        # order of angels is shoulder, hip,knee and each row represents a leg
        self.Joint_Ang = np.zeros((4, 3))
        # Initialization of the The foot positions
        self.F_pos = np.zeros((4, 4))  # this is in body frame note that effectively this is a 4X3 array last cullum is zeros

        # The following array contains the tarnslation needed to construct the homogeneous transformation
        # each row corresponds to a leg in order (LF,RF,LR,RR)
        # Body Shoulder translation
        self.Tr01 = np.array([[self.B_L / 2, self.B_W / 2, 0],
                              [self.B_L / 2, -self.B_W / 2, 0],
                              [-self.B_L / 2, self.B_W / 2, 0],
                              [-self.B_L / 2, -self.B_W / 2, 0]])
        # Shoulder to hip translation
        self.Tr12 = np.array([[0, self.H_L, 0],
                              [0, -self.H_L, 0],
                              [0, self.H_L, 0],
                              [0, -self.H_L, 0]])
        # Hip to knee translation
        self.Tr23 = np.array([[0, 0, -self.T_L],
                              [0, 0, -self.T_L],
                              [0, 0, -self.T_L],
                              [0, 0, -self.T_L]])
    # The following transformation returns both the Homogeneous transformations and their respected Jacobian
    def T01_fun(self, ii):
        T01 = np.array([[1, 0, 0, self.Tr01[ii, 0]],
                        [0, np.cos(self.Joint_Ang[ii, 0]), -np.sin(self.Joint_Ang[ii, 0]), self.Tr01[ii, 1]],
                        [0, np.sin(self.Joint_Ang[ii, 0]), np.cos(self.Joint_Ang[ii, 0]), self.Tr01[ii, 2]],
                        [0, 0, 0, 1]])

        DT01 = np.array([[0, 0, 0, 0],
                         [0, -np.sin(self.Joint_Ang[ii, 0]), -np.cos(self.Joint_Ang[ii, 0]), 0],
                         [0, np.cos(self.Joint_Ang[ii, 0]), -np.sin(self.Joint_Ang[ii, 0]), 0],
                         [0, 0, 0, 0]])
        return T01, DT01

    def T12_fun(self, ii):
        T12 = np.array([[np.cos(self.Joint_Ang[ii, 1]), 0, np.sin(self.Joint_Ang[ii, 1]), self.Tr12[ii, 0]],
                        [0, 1, 0, self.Tr12[ii, 1]],
                        [-np.sin(self.Joint_Ang[ii, 1]), 0, np.cos(self.Joint_Ang[ii, 1]), self.Tr12[ii, 2]],
                        [0, 0, 0, 1]])

        DT12 = np.array([[-np.sin(self.Joint_Ang[ii, 1]), 0, np.cos(self.Joint_Ang[ii, 1]), 0],
                         [0, 0, 0, 0],
                         [-np.cos(self.Joint_Ang[ii, 1]), 0, -np.sin(self.Joint_Ang[ii, 1]), 0],
                         [0, 0, 0, 0]])
        return T12, DT12

    def T23_fun(self, ii):
        T23 = np.array([[np.cos(self.Joint_Ang[ii, 2]), 0, np.sin(self.Joint_Ang[ii, 2]), self.Tr23[ii, 0]],
                        [0, 1, 0, self.Tr23[ii, 1]],
                        [-np.sin(self.Joint_Ang[ii, 2]), 0, np.cos(self.Joint_Ang[ii, 2]), self.Tr23[ii, 2]],
                        [0, 0, 0, 1]])

        DT23 = np.array([[-np.sin(self.Joint_Ang[ii, 2]), 0, np.cos(self.Joint_Ang[ii, 2]), 0],
                         [0, 0, 0, 0],
                         [-np.cos(self.Joint_Ang[ii, 2]), 0, -np.sin(self.Joint_Ang[ii, 2]), 0],
                         [0, 0, 0, 0]])
        return T23, DT23

    def Forw_kin(self, ii):
        # self.set_actuators_for_joint_ang(ii)
        # Computation of the transformation
        T01, DT01 = self.T01_fun(ii)
        T12, DT12 = self.T12_fun(ii)
        T23, DT23 = self.T23_fun(ii)

        self.F_pos[ii] = np.matmul(T01, np.matmul(T12, np.matmul(T23, np.array([0, 0, -self.S_L, 1]))))

    def Inv_Kin(self, ii, DX):
        # DX is a three dimentional vector denoting the required movment of the foot ii

        XF = np.array([self.F_pos[ii, 0] + DX[0] - self.Tr01[ii, 0], self.F_pos[ii, 1] + DX[1] - self.Tr01[ii, 1],
                       self.F_pos[ii, 2] + DX[2] - self.Tr01[ii, 2]])

        l1 = self.H_L
        l2 = self.T_L
        l3 = self.S_L
        XF_n = np.linalg.norm(XF)

        H = np.sqrt(XF_n ** 2 - l1 ** 2)
        t3 = -np.arccos((H ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3))
        t3_a = np.arcsin(l3 * np.sin(t3) / H)
        t2_a = np.arcsin(XF[0] / H)
        t2 = -(t3_a + t2_a)
        H1 = np.sqrt((H * np.cos(t2_a)) ** 2 + l1 ** 2)

        if ii in [1, 3]:
            t1_a = np.arcsin(l1 / H1)
            t1_b = np.arcsin(-XF[1] / H1)
            t1 = -t1_b + t1_a
        elif ii in [0, 2]:
            t1_a = np.arcsin(l1 / H1)
            t1_b = np.arcsin(XF[1] / H1)
            t1 = t1_b - t1_a
        # rospy.loginfo("t1=%f"%t1)
        self.Joint_Ang[ii, 0] = t1
        self.Joint_Ang[ii, 1] = t2
        self.Joint_Ang[ii, 2] = t3
        self.Forw_kin(ii)
