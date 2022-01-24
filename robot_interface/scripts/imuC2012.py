# This file contains the basic functions needed for the imuC2012 consistent leg odomotry

import numpy as np
from scipy import linalg
from q_functions import w2zq, q_mult,skew_m
g=np.array([0,0,9.8])

def Gamma_func(dt,w):
    W=skew_m(w)
    Wi=numpy.linalg.inv(W)
    a0 = np.eye(3)
    a1 = dt * W
    a2 = 0.5*np.matmul(a1,a1)
    G0 = linalg.expm(dt*W)
    G1 = G0-a0
    G2 = np.matmul(G1-a1,Wi)
    G3 = np.matmul(G1-a1-a2,np.matmul(Wi,Wi))
    return G0, G1, G2, G3


def F_func(dt,f,C,w,Qf,Qbf,Qw,Qbw):
    f_cross=skew_m(f)
    G0, G1, G2, G3=Gamma_func(dt,w)
    F=np.array([[np.eye(3)      , dt*np.eye(3)   ,  -0.5*dt*dt*np.matmul(C,f_cross), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -0.5*dt*dt*C   , np.zeros([3,3])],  #dr
                [np.zeros([3,3]), np.eye(3)      ,   -dt*np.matmul(C,f_cross)      , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), dt*C           , np.zeros([3,3])],  #dv
                [np.zeros([3,3]), np.zeros([3,3]),   G0.tarnspose()                            , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -G1.transpose()             ],  #dq
                [np.zeros([18,9]), np.eye(18)]])

    Q=np.array([[((dt**3)/3)*Qf+((dt**5)/20)*Qbf, ((dt**2)/2)*Qf+((dt**4)/8)*Qbf, np.zeros([3,3])                       , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -((dt**3)/6)*Qbf, np.zeros([3,3])],#dr
                [((dt**2)/2)*Qf+((dt**4)/8)*Qbf , dt*Qf+((dt**3)/3)*Qbf         , np.zeros([3,3])                       , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -((dt**3)/6)*Qbf, np.zeros([3,3])],#dv
                [zeros([3,3])                   , zeros([3,3])                  ,dt*Qw+np.matmul(G3+G3.transpose(),Qbw) , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]) ,-np.matmul(G2.transpose(),Qbw)],#q
                [zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(Qf, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])],
                [zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), np.zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(Qf, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])],
                [zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(Qf, C)), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])],
                [zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt * np.matmul(C.transpose(), np.matmul(Qf, C)), np.zeros([3, 3]), np.zeros([3, 3])],
                [-(dt**3)/6*np.matmul(Qbf, C), -(dt**2)/2*np.matmul(Qbf, C),np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3]), dt*Qbf,np.zeros([3, 3])],
                [zeros([3, 3]), zeros([3, 3]), -np.matmul(Qbw, G2),zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), zeros([3, 3]), dt*Qbw]])

    return F, Q


def H_func(C,r,p1,p2,p3,p4,s1,s2,s3,s4):
    H=np.array([[-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), C, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)],
               [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), C, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)],
               [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), np.zeros(3), C, np.zeros(3), np.zeros(3), np.zeros(3)],
               [-C, np.zeros(3), skew_m(np.matmul(C,p1-r)), np.zeros(3), np.zeros(3), np.zeros(3), C, np.zeros(3), np.zeros(3)]])
    y=np.array([[s1-np.matmul(C,p1-r)],
               [s2-np.matmul(C,p2-r)],
               [s3-np.matmul(C,p3-r)],
               [s4-np.matmul(C,p4-r)]])

    return H, y

