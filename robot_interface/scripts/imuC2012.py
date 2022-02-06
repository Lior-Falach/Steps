# This file contains the basic functions needed for the imuC2012 consistent leg odomotry
import rospy
import numpy as np
#import numpy.linalg as linalg
from scipy import linalg
from q_functions import w2zq, q_mult,skew_m
g=np.array([0,0,9.8])

def Gamma_func(dt,w):
    W = skew_m(w)
    Wi = W #np.linalg.inv(W) #Need to correct this
    a0 = np.eye(3)
    a1 = dt * W
    a2 = 0.5*np.matmul(a1,a1)
    G0 = linalg.expm(dt*W)
    G1 = G0-a0
    G2 = np.matmul(G1-a1,Wi)
    G3 = np.matmul(G1-a1-a2,np.matmul(Wi,Wi))
    return G0, G1, G2, G3


def F_func(dt,f,C,w,Qf,Qbf,Qw,Qbw,Qp,contact):
    f_cross=skew_m(f)
    G0, G1, G2, G3=Gamma_func(dt,w)
    high=1e6
    low=1
    n1 = high * contact[0] + low
    n2 = high * contact[1] + low
    n3 = high * contact[2] + low
    n4 = high * contact[3] + low
    F1 = np.concatenate((np.eye(3)      , dt*np.eye(3)   ,  -0.5*dt*dt*np.matmul(C,f_cross), np.zeros([3,3]),
                         np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -0.5*dt*dt*C   , np.zeros([3,3])), axis=1)
    F2 = np.concatenate((np.zeros([3, 3]), np.eye(3), -dt * np.matmul(C, f_cross), np.zeros([3, 3]),
                         np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt * C, np.zeros([3, 3])),axis=1)  # dv
    F3 = np.concatenate((np.zeros([3,3]), np.zeros([3,3]),   G0.transpose()                , np.zeros([3,3]),
                         np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -G1.transpose()),axis=1)
    F4 = np.concatenate((np.zeros([18,9]), np.eye(18)),axis=1)
        # rospy.loginfo(F1)
    F = np.concatenate((F1,F2,F3,F4),axis=0)


    Q1 = np.concatenate((((dt**3)/3)*Qf+((dt**5)/20)*Qbf, ((dt**2)/2)*Qf+((dt**4)/8)*Qbf, np.zeros([3,3]),
                         np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -((dt**3)/6)*Qbf, np.zeros([3,3])),axis=1)#dr
    Q2 = np.concatenate((((dt**2)/2)*Qf+((dt**4)/8)*Qbf , dt*Qf+((dt**3)/3)*Qbf, np.zeros([3,3]),
                         np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -((dt**3)/6)*Qbf, np.zeros([3,3])),axis=1)#dv
    Q3 = np.concatenate((np.zeros([3,3]), np.zeros([3,3]),dt*Qw+np.matmul(G3+G3.transpose(),Qbw) ,
                         np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]) ,-np.matmul(G2.transpose(),Qbw)),axis=1)#q
    Q4 = np.concatenate((np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt * np.matmul(C.transpose(),
                         np.matmul(n1 * Qp, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])),axis=1)
    Q5 = np.concatenate((np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]),
                         dt * np.matmul(C.transpose(), np.matmul(n2 * Qp, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])),axis=1)
    Q6 = np.concatenate((np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]),
                         np.zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(n3 * Qp, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])),axis=1)
    Q7 = np.concatenate((np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]),
                         np.zeros([3, 3]), np.zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(n4 * Qp, C)), np.zeros([3, 3]), np.zeros([3, 3])),axis=1)
    Q8 = np.concatenate((-(dt**3)/6*np.matmul(Qbf, C), -(dt**2)/2*np.matmul(Qbf, C),np.zeros([3, 3]),
                         np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt*Qbf,np.zeros([3, 3])),axis=1)
    Q9 = np.concatenate((np.zeros([3, 3]), np.zeros([3, 3]), -np.matmul(Qbw, G2),np.zeros([3, 3]),
                         np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt*Qbw),axis=1)

    Q = np.concatenate((Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9),axis=0)

    return F, Q


def H_func(C,r,p1,p2,p3,p4,s1,s2,s3,s4):

    H1 = np.concatenate((-C, np.zeros([3,3]), skew_m(np.matmul(C, p1 - r)), C, np.zeros([3,15])),axis=1)
    # rospy.loginfo(H1)
    H2 = np.concatenate((-C, np.zeros([3,3]), skew_m(np.matmul(C, p2 - r)), C, np.zeros([3,15])), axis=1)
    # rospy.loginfo(H2)
    H3 = np.concatenate((-C, np.zeros([3,3]), skew_m(np.matmul(C, p3 - r)), C, np.zeros([3,15])), axis=1)
    # rospy.loginfo(H3)
    H4 = np.concatenate((-C, np.zeros([3,3]), skew_m(np.matmul(C, p4 - r)), C, np.zeros([3,15])), axis=1)
    # rospy.loginfo(H4)
    H = np.concatenate((H1, H2, H3, H4),axis=0)
    # H=np.array([[-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), C, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)],
    #            [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), C, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)],
    #            [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), np.zeros(3), C, np.zeros(3), np.zeros(3), np.zeros(3)],
    #            [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), np.zeros(3), np.zeros(3), C, np.zeros(3), np.zeros(3)]])
    y = np.concatenate( (s1-np.matmul(C,p1-r),s2-np.matmul(C,p2-r),s3-np.matmul(C,p3-r),s4-np.matmul(C,p4-r)),axis=0)

    return H, y

