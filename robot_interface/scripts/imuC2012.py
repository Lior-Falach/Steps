# This file contains the basic functions needed for the imuC2012 consistent leg odomotry

import numpy as np
from q_functions import w2zq, q_mult
g=np.array([0,0,9.8])

P_0=


def f_func(x,dt,C,f,w):
    g=[0,0,-9.8]
    r=x[0:3]
    v=x[3:6]
    q=x[6:10]
    p1 = x[10:13]
    p2 = x[13:16]
    p3 = x[16:19]
    p4 = x[19:22]
    bf = x[22:25]
    bw = x[25:28]
    r_pri=r+dt*v+0.5*dt*dt*(np.matmul(C,f-bf)+g)
    v_pri=v+dt*(np.matmul(C,f-bf)+g)
    q_pri=q_mult(w2zq(dt*(w-bw)),q)

    return np.array([r_pri,v_pri,q_pri,p1,p2,p3,p4,bf,bw])


    return

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
