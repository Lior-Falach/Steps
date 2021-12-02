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
from imuC2012 import f_func, h_func,F_func, H_func, Q_0, R_0


class Est(object):

    def __init__(self, x_0, P_0,):# Q_0, R_0 ,f_func , h_func, F_func, H_func ):
        "type is a string whihc can take either 'EKF' or 'UKF'"
        "x_0 and P_0 are the initial state and state Covariance matrices numpy.array(dim_x,1) numpy.array(dim_x, dim_x)"
        "Q_0 and R_0 are rge initial Process and Measurment Covariance Noise numpy.array(dim_x, dim_x) numpy.array(dim_z, dim_z)"
        #The following four arguments should reffer to function
        "f is a handle to the system dynamics x_k+1=f(x_k,u_k)" #x=f_func(x,u)
        "h is the measurment function z_k=h(x_k,u_k)" #z=h_func(x,u)
        "F is the gradient of f w.r.t. x" #F=F_func(x)
        "H is the gradiant of h w.r.t. x" #H=H_func(x)
        self.x_post=x_0
        self.x_prior=x_0
        self.P_post=P_0
        self.P_prior=P_0
        self.Type=type
        self.Q=Q_0
        self.R=R_0
        self.f_func=f_func
        self.h_func = h_func
        self.F_func = F_func
        self.H_func = H_func
        self.K=[]#"Initialize the Kalman Gain"

    def predict_state(self, u):
         "Performs the state prediction innovation of the extended Kalman"
        self.x_prior=self.f_func(self.x_post,u)
    def predict_cov(self):
        if self.Type=='EKF':
            F = self.F_func(self.x_post)
            self.P_prior = dot(F, self.P_postP).dot(F.T) + self.Q
        if self.Type=='UKF':
            "Compute Sigma Points"
            "Predict Sigma Points"
            "Compute predicted Covariance"
    def comu_K_gain(self):
        H=self.H_func(self.x_prior)
        S=dot(H, self.P_postP).dot(H.T) + self.R
        self.K= dot(self.P_prior, H.T).dot(linalg.inv(self.S))
    def update_state(self,z):
        self.x_post=self.x_prior+dot(self.K,)


