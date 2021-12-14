# This file contains the basic functions needed for the imuC2012 consistent leg odomotry

import numpy as np
from q_functions import w2zq, q_mult,skew_m
g=np.array([0,0,9.8])

P_0=


def F_func(dt,f,C,w):
    f_cross=skew_m(f)
    G0,G1,G2,G3=Gamma_func(dt,w)
    F=np.array([[np.eye(3)      , dt*np.eye(3)   ,  -0.5*dt*dt*np.matmul(C,f_cross), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), -0.5*dt*dt*C   , np.zeros([3,3])],  #dr
                [np.zeros([3,3]), np.eye(3)      ,   -dt*np.matmul(C,f_cross)      , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), dt*C           , np.zeros([3,3])],  #dv
                [np.zeros([3,3]), np.zeros([3,3]),   G0                            , np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), np.zeros([3,3]), G1             ],  #dq
                [np.zeros([18,9]), np.eye(18)])

    return F


def h_func():

    return

def F_func():

    return

def H_func():

    return

def R_0():

    return

def Q_0():

    return
