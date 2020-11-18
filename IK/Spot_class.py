# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:42:43 2020

@author: liorfa
"""
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d
# Import the PCA9685 module.
#import Adafruit_PCA9685


# import random 

#Transformation constant

T_DR=np.pi/180
T_RS=500/np.pi

# Defining the spot class
class Spot:
    def __init__(self, Shin_L,Thigh_L,Hip_L,Body_W,Body_L):
        # fig = plt.figure()
        # self.ax = plt.axes(projection='3d')
        
        #Initialization of the servo expantion board
        #self.pwm=Adafruit_PCA9685.PCA9685()
        #self.pwm.set_pwm_freq(60)
        
        # step size for iterative inverse kinematic calculation
        self.S_inc=0.01 #defined in meters
        
        #Initialized position
        self.pos=np.array([0,0,0])
        # The continuous walking gates
        self.Walk_D=0.05
        self.Walk_L=0.03
        self.Walk_Se=[0,3,1,2] #sequence of walking legs
        
        self.Current_swing=5
        self.Vel_Dir=np.array([0,0,0]) # A three dimentional vecote indicating the referance direction of walking
        self.Goal=np.array([0.5,0,0])
        
        #these are the linkage length of the robot, in general they should be fixed
        self.S_L = Shin_L
        self.T_L = Thigh_L
        self.H_L=Hip_L
        self.B_W=Body_W
        self.B_L=Body_L
        
        # # Base angles---Home position     
        # self.Servo_Cmd=0*np.array([250,530,440,480,230,270,290,500,490,480,200,320])
        self.Joint_Ang=np.array([180,-145,10,180,-145,-10,180,-145,10,180,-145,-10])*T_DR

        # #Base values
        # self.Servo_Cmd_base=np.array([270,520,430,450,250,270,290,530,450,470,200,300])
        # self.Joint_Ang_base=np.array([90 ,-90,0  , 90,-90,0  ,90,-90 ,0  ,90,-90 ,0])*T_DR
        # self.Direction=np.array([1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1])
        
        # Initialize the positions of the joints w.r.t. the initial angles 
        self.LF_Forw_kin()
        self.RF_Forw_kin()
        self.LR_Forw_kin()
        self.RR_Forw_kin()
        self.Feet_array()
       
        # #Initializing the motor commands 
        # for i in range(0,4,1):
        #     self.Servo_Cmd_update(i)
        
        
        # # Workspace parameters
        # Xp=0.2
        # Xn=-0.2
        # Yp=0.2
        # Yn=-0.28
        # Zp=0.25
        # Zn=-0.25
        # #Setting the workspace for each leg
        # # LF
        # Up=np.array([[Xp,Yp,Zp],[Xp,-Yn,Zp],[Xp,Yp,Zp],[Xp,-Yn,Zp]])
        # Down=np.array([[Xn,Yn,Zn],[Xn,-Yp,Zn],[Xn,Yn,Zn],[Xn,-Yp,Zn]])
        # self.Upper_bnd=Up+self.F_pos
        # self.Lower_bnd=Down+self.F_pos
        
        # self.LF_bound_X=np.array([Xn,Xp])+self.LF_F_pos[0]
        # self.LF_bound_Y=np.array([Yn,Yp])+self.LF_F_pos[1]
        # self.LF_bound_Z=np.array([Zn,Zp])+self.LF_F_pos[2]
        # # RF  
        # self.RF_bound_X=np.array([Xn,Xp])+self.RF_F_pos[0]
        # self.RF_bound_Y=-np.array([Yn,Yp])+self.RF_F_pos[1]
        # self.RF_bound_Z=np.array([Zn,Zp])+self.RF_F_pos[2]
        # # LR
        # self.LR_bound_X=np.array([Xn,Xp])+self.LR_F_pos[0]
        # self.LR_bound_Y=np.array([Yn,Yp])+self.LR_F_pos[1]
        # self.LR_bound_Z=np.array([Zn,Zp])+self.LR_F_pos[2]
        # # RR
        # self.RR_bound_X=np.array([Xn,Xp])+self.RR_F_pos[0]
        # self.RR_bound_Y=-np.array([Yn,Yp])+self.RR_F_pos[1]
        # self.RR_bound_Z=np.array([Zn,Zp])+self.RR_F_pos[2]
    
            
        self.LF_bound_check=np.array([0,0,0])
        self.RF_bound_check=np.array([0,0,0])
        self.LR_bound_check=np.array([0,0,0])
        self.RR_bound_check=np.array([0,0,0])
    def Feet_array(self):
        self.F_pos=np.array([self.LF_F_pos[0:3],self.RF_F_pos[0:3],self.LR_F_pos[0:3],self.RR_F_pos[0:3]])
        
    # def Servo_Cmd_update(self,m):
    #     for i in range(m*3,3*(m+1),1):
            
    #         self.Servo_Cmd[i]=self.Servo_Cmd_base[i]+T_RS*self.Direction[i]*(self.Joint_Ang[i]-(self.Joint_Ang_base[i]))
    #         self.pwm.set_pwm(i, 0, int(self.Servo_Cmd[i]))
    #         time.sleep(0.1)
    #         #print(i)
    #         #print(self.Joint_Ang[i]/T_DR)
    #         #print(self.Servo_Cmd[i])
            
    
        
    #self.Body_Attitude(0,0,0)
    # def check_workspace(self,D,leg):
    #     X=self.F_pos[leg,:]+D #progected pos
    #     Up=self.Upper_bnd[leg,:]
    #     Down=self.Lower_bnd[leg,:]
    #     print(X)
    #     print(Up)
    #     print(Down)
    #     DDx=np.array([0,0,0])
    #     u=Up-X
    #     u[u>0]=0
    #     d=X-Down
    #     d[d>0]=0
    #     DDx=u-d
    #     print(DDx)
    #     c=sum(DDx!=0)
    #     return [DDx,c];
      
        
    def LF_Forw_kin(self):
        
        #Soulder transformatiom
        T01 =np.array([[1,0,0,self.B_L/2],[0,np.cos(self.Joint_Ang[2]),-np.sin(self.Joint_Ang[2]),self.B_W/2],[0,np.sin(self.Joint_Ang[2]),np.cos(self.Joint_Ang[2]),0],[0,0,0,1]])
        #Hip transformation
        T12 =np.array([[np.cos(self.Joint_Ang[1]),0,np.sin(self.Joint_Ang[1]),0],[0,1,0,self.H_L],[-np.sin(self.Joint_Ang[1]),0,np.cos(self.Joint_Ang[1]),0],[0,0,0,1]])
        #knee transformation
        T23 =np.array([[np.cos(self.Joint_Ang[0]),0,np.sin(self.Joint_Ang[0]),-self.T_L],[0,1,0,0],[-np.sin(self.Joint_Ang[0]),0,np.cos(self.Joint_Ang[0]),0],[0,0,0,1]])
        
        #The Jackobian LF
        DT01=np.array([[0,0,0,0],[0,-np.sin(self.Joint_Ang[2]),-np.cos(self.Joint_Ang[2]),0],[0,np.cos(self.Joint_Ang[2]),-np.sin(self.Joint_Ang[2]),0],[0,0,0,0]])
        DT12=np.array([[-np.sin(self.Joint_Ang[1]),0,np.cos(self.Joint_Ang[1]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[1]),0,-np.sin(self.Joint_Ang[1]),0],[0,0,0,0]])
        DT23=np.array([[-np.sin(self.Joint_Ang[0]),0,np.cos(self.Joint_Ang[0]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[0]),0,-np.sin(self.Joint_Ang[0]),0],[0,0,0,0]])
        J1=np.matmul(DT01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J2=np.matmul(T01,np.matmul(DT12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J3=np.matmul(T01,np.matmul(T12,np.matmul(DT23,np.array([0,0,-self.S_L,1]))))
        
        J=np.array([J3,J2,J1])
        J=np.transpose(J)
        J=J[0:3,0:3]
        self.LF_J=J
        self.LF_inv_J=np.linalg.inv(J)
       
        # shoulder location
        self.LF_S_pos=np.matmul(T01,np.matmul(T12,np.array([0,0,0,1])))
        # knee location
        self.LF_K_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,0,1]))))
        #foot location
        self.LF_F_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        
        #self.ax.plot3D([self.LF_F_pos[0]],[self.LF_F_pos[1]],[self.LF_F_pos[2]],'k*')
       
    def LF_Inv_kin(self,DX):
        # [DDx,c]=self.check_workspace(DX,0)
        # DX=DX+DDx
        # self.LF_WC=c
        
        # Partition of the step size by the predetermined increment
        XF=np.array([self.LF_F_pos[0]+DX[0],self.LF_F_pos[1]+DX[1],self.LF_F_pos[2]+DX[2]])
        S=np.sqrt(DX.dot(DX))
        step_num=int(S/self.S_inc)+1
        dx=DX/step_num;
        
        for i in range(0,step_num,1):
            X=np.array([self.LF_F_pos[0],self.LF_F_pos[1],self.LF_F_pos[2]])
            dx=(XF-X)/(step_num-i)
            #print(dx)
            DT=np.matmul(self.LF_inv_J,dx)
            #print(DT)
            self.Joint_Ang[2]+=DT[2]
            self.Joint_Ang[1]+=DT[1]
            self.Joint_Ang[0]+=DT[0]
            self.LF_Forw_kin()
        
        #print([self.Joint_Ang[0]/T_DR,self.Joint_Ang[1]/T_DR,self.Joint_Ang[2]/T_DR])
        # self.Servo_Cmd_update(0)

     
    def RF_Forw_kin(self):        
        #Soulder transformatiom
        T01=np.array([[1,0,0,self.B_L/2],[0,np.cos(self.Joint_Ang[5]),-np.sin(self.Joint_Ang[5]),-self.B_W/2],[0,np.sin(self.Joint_Ang[5]),np.cos(self.Joint_Ang[5]),0],[0,0,0,1]])
        #Hip transformation
        T12=np.array([[np.cos(self.Joint_Ang[4]),0,np.sin(self.Joint_Ang[4]),0],[0,1,0,-self.H_L],[-np.sin(self.Joint_Ang[4]),0,np.cos(self.Joint_Ang[4]),0],[0,0,0,1]])
        #knee transformation
        T23=np.array([[np.cos(self.Joint_Ang[3]),0,np.sin(self.Joint_Ang[3]),-self.T_L],[0,1,0,0],[-np.sin(self.Joint_Ang[3]),0,np.cos(self.Joint_Ang[3]),0],[0,0,0,1]])
        
        
        #The Jackobian
        DT01=np.array([[0,0,0,0],[0,-np.sin(self.Joint_Ang[5]),-np.cos(self.Joint_Ang[5]),0],[0,np.cos(self.Joint_Ang[5]),-np.sin(self.Joint_Ang[5]),0],[0,0,0,0]])
        DT12=np.array([[-np.sin(self.Joint_Ang[4]),0,np.cos(self.Joint_Ang[4]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[4]),0,-np.sin(self.Joint_Ang[4]),0],[0,0,0,0]])
        DT23=np.array([[-np.sin(self.Joint_Ang[3]),0,np.cos(self.Joint_Ang[3]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[3]),0,-np.sin(self.Joint_Ang[3]),0],[0,0,0,0]])
        J1=np.matmul(DT01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J2=np.matmul(T01,np.matmul(DT12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J3=np.matmul(T01,np.matmul(T12,np.matmul(DT23,np.array([0,0,-self.S_L,1]))))
        
        J=np.array([J3,J2,J1])
        # print(J)
        J=np.transpose(J)
        J=J[0:3,0:3]
        self.RF_J=J
        self.RF_inv_J=np.linalg.inv(J)
        
        # shoulder location
        self.RF_S_pos=np.matmul(T01,np.matmul(T12,np.array([0,0,0,1])))
        # knee location
        self.RF_K_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,0,1]))))
        #foot location
        self.RF_F_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        #self.ax.plot3D([self.RF_F_pos[0]],[self.RF_F_pos[1]],[self.RF_F_pos[2]],'b*')

    def RF_Inv_kin(self,DX):
        # [DDx,c]=self.check_workspace(DX,1)
        # DX=DX+DDx
        # self.RF_WC=c
        # Partition of the step size by the predetermined increment
        XF=np.array([self.RF_F_pos[0]+DX[0],self.RF_F_pos[1]+DX[1],self.RF_F_pos[2]+DX[2]])
        S=np.sqrt(DX.dot(DX))
        step_num=int(S/self.S_inc)+1
        dx=DX/step_num;
        
        for i in range(0,step_num,1):
            X=np.array([self.RF_F_pos[0],self.RF_F_pos[1],self.RF_F_pos[2]])
            dx=(XF-X)/(step_num-i)
            DT=np.matmul(self.RF_inv_J,dx)
            self.Joint_Ang[5]+=DT[2]
            self.Joint_Ang[4]+=DT[1]
            self.Joint_Ang[3]+=DT[0]
            self.RF_Forw_kin()
        
        # self.Servo_Cmd_update(1)
        
    def LR_Forw_kin(self):
        
        #Soulder transformatiom
        T01=np.array([[1,0,0,-self.B_L/2],[0,np.cos(self.Joint_Ang[8]),-np.sin(self.Joint_Ang[8]),self.B_W/2],[0,np.sin(self.Joint_Ang[8]),np.cos(self.Joint_Ang[8]),0],[0,0,0,1]])
        #Hip transformation
        T12=np.array([[np.cos(self.Joint_Ang[7]),0,np.sin(self.Joint_Ang[7]),0],[0,1,0,self.H_L],[-np.sin(self.Joint_Ang[7]),0,np.cos(self.Joint_Ang[7]),0],[0,0,0,1]])
        #knee transformation
        T23=np.array([[np.cos(self.Joint_Ang[6]),0,np.sin(self.Joint_Ang[6]),-self.T_L],[0,1,0,0],[-np.sin(self.Joint_Ang[6]),0,np.cos(self.Joint_Ang[6]),0],[0,0,0,1]])
         #The Jackobian
        DT01=np.array([[0,0,0,0],[0,-np.sin(self.Joint_Ang[8]),-np.cos(self.Joint_Ang[8]),0],[0,np.cos(self.Joint_Ang[8]),-np.sin(self.Joint_Ang[8]),0],[0,0,0,0]])
        DT12=np.array([[-np.sin(self.Joint_Ang[7]),0,np.cos(self.Joint_Ang[7]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[7]),0,-np.sin(self.Joint_Ang[7]),0],[0,0,0,0]])
        DT23=np.array([[-np.sin(self.Joint_Ang[6]),0,np.cos(self.Joint_Ang[6]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[6]),0,-np.sin(self.Joint_Ang[6]),0],[0,0,0,0]])
        J1=np.matmul(DT01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J2=np.matmul(T01,np.matmul(DT12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J3=np.matmul(T01,np.matmul(T12,np.matmul(DT23,np.array([0,0,-self.S_L,1]))))
        J=np.array([J3,J2,J1])
        # print(J)
        J=np.transpose(J)
        J=J[0:3,0:3]
        self.LR_J=J
        self.LR_inv_J=np.linalg.inv(J)
        
        # shoulder location
        self.LR_S_pos=np.matmul(T01,np.matmul(T12,np.array([0,0,0,1])))
        # knee location
        self.LR_K_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,0,1]))))
        #foot location
        self.LR_F_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        #self.ax.plot3D([self.LR_F_pos[0]],[self.LR_F_pos[1]],[self.LR_F_pos[2]],'g*')
        
    def LR_Inv_kin(self,DX):
        # [DDx,c]=self.check_workspace(DX,2)
        # DX=DX+DDx
        # self.LR_WC=c
        # Partition of the step size by the predetermined increment
        XF=np.array([self.LR_F_pos[0]+DX[0],self.LR_F_pos[1]+DX[1],self.LR_F_pos[2]+DX[2]])
        S=np.sqrt(DX.dot(DX))
        step_num=int(S/self.S_inc)+1
        dx=DX/step_num;
        
        for i in range(0,step_num,1):
            X=np.array([self.LR_F_pos[0],self.LR_F_pos[1],self.LR_F_pos[2]])
            dx=(XF-X)/(step_num-i)
            DT=np.matmul(self.LR_inv_J,dx)
            self.Joint_Ang[8]+=DT[2]
            self.Joint_Ang[7]+=DT[1]
            self.Joint_Ang[6]+=DT[0]
            self.LR_Forw_kin()
        
        # self.Servo_Cmd_update(2)
        
    def RR_Forw_kin(self):
        
        #Soulder transformatiom
        T01=np.array([[1,0,0,-self.B_L/2],[0,np.cos(self.Joint_Ang[11]),-np.sin(self.Joint_Ang[11]),-self.B_W/2],[0,np.sin(self.Joint_Ang[11]),np.cos(self.Joint_Ang[11]),0],[0,0,0,1]])
        #Hip transformation
        T12=np.array([[np.cos(self.Joint_Ang[10]),0,np.sin(self.Joint_Ang[10]),0],[0,1,0,-self.H_L],[-np.sin(self.Joint_Ang[10]),0,np.cos(self.Joint_Ang[10]),0],[0,0,0,1]])
        #knee transformation
        T23=np.array([[np.cos(self.Joint_Ang[9]),0,np.sin(self.Joint_Ang[9]),-self.T_L],[0,1,0,0],[-np.sin(self.Joint_Ang[9]),0,np.cos(self.Joint_Ang[9]),0],[0,0,0,1]])
        #The Jackobian
        DT01=np.array([[0,0,0,0],[0,-np.sin(self.Joint_Ang[11]),-np.cos(self.Joint_Ang[11]),0],[0,np.cos(self.Joint_Ang[11]),-np.sin(self.Joint_Ang[11]),0],[0,0,0,0]])
        DT12=np.array([[-np.sin(self.Joint_Ang[10]),0,np.cos(self.Joint_Ang[10]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[10]),0,-np.sin(self.Joint_Ang[10]),0],[0,0,0,0]])
        DT23=np.array([[-np.sin(self.Joint_Ang[9]),0,np.cos(self.Joint_Ang[9]),0],[0,0,0,0],[-np.cos(self.Joint_Ang[9]),0,-np.sin(self.Joint_Ang[9]),0],[0,0,0,0]])
        J1=np.matmul(DT01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J2=np.matmul(T01,np.matmul(DT12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        J3=np.matmul(T01,np.matmul(T12,np.matmul(DT23,np.array([0,0,-self.S_L,1]))))
        J=np.array([J3,J2,J1])
        # print(J)
        J=np.transpose(J)
        J=J[0:3,0:3]
        self.RR_J=J
        self.RR_inv_J=np.linalg.inv(J)
        
        # shoulder location
        self.RR_S_pos=np.matmul(T01,np.matmul(T12,np.array([0,0,0,1])))
        # knee location
        self.RR_K_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,0,1]))))
        #foot location
        self.RR_F_pos=np.matmul(T01,np.matmul(T12,np.matmul(T23,np.array([0,0,-self.S_L,1]))))
        #self.ax.plot3D([self.RR_F_pos[0]],[self.RR_F_pos[1]],[self.RR_F_pos[2]],'r*')

    def RR_Inv_kin(self,DX):
        # [DDx,c]=self.check_workspace(DX,3)
        # DX=DX+DDx
        # self.RR_WC=c
         # Partition of the step size by the predetermined increment
        XF=np.array([self.RR_F_pos[0]+DX[0],self.RR_F_pos[1]+DX[1],self.RR_F_pos[2]+DX[2]])
        S=np.sqrt(DX.dot(DX))
        step_num=int(S/self.S_inc)+1
        dx=DX/step_num;
        
        for i in range(0,step_num,1):
            X=np.array([self.RR_F_pos[0],self.RR_F_pos[1],self.RR_F_pos[2]])
            dx=(XF-X)/(step_num-i)
            DT=np.matmul(self.RR_inv_J,dx)
            self.Joint_Ang[11]+=DT[2]
            self.Joint_Ang[10]+=DT[1]
            self.Joint_Ang[9]+=DT[0]
            self.RR_Forw_kin()
        
        # self.Servo_Cmd_update(3)
       
        
    # def Body_Attitude(self,roll,pitch,yaw):
        
    #     #roll transformatiom
    #     T_roll=np.array([[1,0,0,0],[0,np.cos(roll),np.sin(roll),0],[0,-np.sin(roll),np.cos(roll),0],[0,0,0,1]])
    #     #pitch transformation
    #     T_pitch=np.array([[np.cos(pitch),0,-np.sin(pitch),0],[0,1,0,0],[np.sin(pitch),0,np.cos(pitch),0],[0,0,0,1]])
    #     #yaw transformation
    #     T_Yaw=np.array([[np.cos(yaw),np.sin(yaw),0,0],[-np.sin(yaw),np.cos(yaw),0,0],[0,0,1,0],[0,0,0,1]])
    #     #Foot transformation
    #     T=np.matmul(T_Yaw,np.matmul(T_pitch,T_roll))
        
    #     # shoulder locations
    #     self.LF_S_pos=np.matmul(T,self.LF_S_pos)
    #     self.RF_S_pos=np.matmul(T,self.RF_S_pos)
    #     self.LR_S_pos=np.matmul(T,self.LR_S_pos)
    #     self.RR_S_pos=np.matmul(T,self.RR_S_pos)
    #     # knee locations
    #     self.LF_K_pos=np.matmul(T,self.LF_K_pos)
    #     self.RF_K_pos=np.matmul(T,self.RF_K_pos)
    #     self.LR_K_pos=np.matmul(T,self.LR_K_pos)
    #     self.RR_K_pos=np.matmul(T,self.RR_K_pos)
    #     #foot location
    #     self.LF_F_pos=np.matmul(T,self.LF_F_pos)
    #     self.RF_F_pos=np.matmul(T,self.RF_F_pos)
    #     self.LR_F_pos=np.matmul(T,self.LR_F_pos)
    #     self.RR_F_pos=np.matmul(T,self.RR_F_pos)
        
    #This function will alternate the swinf and stance of the feets
    # def Walk_seq_controller(self):
    #     if self.Current_swing==5:
    #         self.Current_swing_ind=0
    #     else:
    #         self.Current_swing_ind+=1
    #         self.Current_swing_ind=self.Current_swing_ind % 4
        
    #     self.Current_swing=self.Walk_Se[self.Current_swing_ind]
            
            
        
    # def Motion_per_leg(self):
    #     #First lift swing leg
    #     DX=np.array([0,0,self.Walk_L])
    #     if self.Current_swing==0:
    #         self.LF_Inv_kin(DX)
    #     elif self.Current_swing==1:
    #         self.RF_Inv_kin(DX)
    #     elif self.Current_swing==2:
    #         self.LR_Inv_kin(DX)
    #     elif self.Current_swing==3:
    #         self.RR_Inv_kin(DX)
            
    #     # Move all legs ih needed direction    
    #     DX=(self.Vel_Dir/np.linalg.norm(self.Vel_Dir))*self.Walk_D
    #     if self.Current_swing==0:
    #         self.LF_Inv_kin(DX)
    #         self.RF_Inv_kin(-DX/3)
    #         self.LR_Inv_kin(-DX/3)
    #         self.RR_Inv_kin(-DX/3)
    #     elif self.Current_swing==1:
    #         self.LF_Inv_kin(-DX/3)
    #         self.RF_Inv_kin(DX)
    #         self.LR_Inv_kin(-DX/3)
    #         self.RR_Inv_kin(-DX/3)
    #     elif self.Current_swing==2:
    #         self.LF_Inv_kin(-DX/3)
    #         self.RF_Inv_kin(-DX/3)
    #         self.LR_Inv_kin(DX)
    #         self.RR_Inv_kin(-DX/3)
    #     elif self.Current_swing==3:
    #         self.LF_Inv_kin(-DX/3)
    #         self.RF_Inv_kin(-DX/3)
    #         self.LR_Inv_kin(-DX/3)
    #         self.RR_Inv_kin(DX)
        
    #     self.pos=np.add(self.pos,DX/3)
    #     #Last place swing leg
    #     DX=np.array([0,0,-self.Walk_L])
    #     if self.Current_swing==0:
    #         self.LF_Inv_kin(DX)
    #     elif self.Current_swing==1:
    #         self.RF_Inv_kin(DX)
    #     elif self.Current_swing==2:
    #         self.LR_Inv_kin(DX)
    #     elif self.Current_swing==3:
    #         self.RR_Inv_kin(DX)
        
    
    # def Tragectory_controller(self):
    #     self.Vel_Dir=self.Goal-self.pos
    #     self.Walk_seq_controller()
    #     self. Motion_per_leg()
        









# # Define the suport functio for plotting the robots pos 
def plot_spot(A,with_box):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    body_x=np.array([A.B_L/2, A.B_L/2, -A.B_L/2, -A.B_L/2, A.B_L/2])
    body_y=np.array([A.B_W/2, -A.B_W/2, -A.B_W/2, A.B_W/2, A.B_W/2])
    body_z=np.array([0,0,0,0,0])
    ax.plot3D(body_x, body_y, body_z, 'gray')
    #LF
    LF_S_x=np.array([A.B_L/2,A.LF_S_pos[0]])
    LF_S_y=np.array([A.B_W/2,A.LF_S_pos[1]])
    LF_S_z=np.array([0,A.LF_S_pos[2]])
    ax.plot3D(LF_S_x, LF_S_y, LF_S_z, 'green')
    LF_H_x=np.array([A.LF_S_pos[0],A.LF_K_pos[0]])
    LF_H_y=np.array([A.LF_S_pos[1],A.LF_K_pos[1]])
    LF_H_z=np.array([A.LF_S_pos[2],A.LF_K_pos[2]])
    ax.plot3D(LF_H_x, LF_H_y, LF_H_z, 'red')
    LF_F_x=np.array([A.LF_K_pos[0],A.LF_F_pos[0]])
    LF_F_y=np.array([A.LF_K_pos[1],A.LF_F_pos[1]])
    LF_F_z=np.array([A.LF_K_pos[2],A.LF_F_pos[2]])
    ax.plot3D(LF_F_x, LF_F_y, LF_F_z, 'blue')
    #RF
    RF_S_x=np.array([A.B_L/2,A.RF_S_pos[0]])
    RF_S_y=np.array([-A.B_W/2,A.RF_S_pos[1]])
    RF_S_z=np.array([0,A.RF_S_pos[2]])
    ax.plot3D(RF_S_x, RF_S_y, RF_S_z, 'green')
    RF_H_x=np.array([A.RF_S_pos[0],A.RF_K_pos[0]])
    RF_H_y=np.array([A.RF_S_pos[1],A.RF_K_pos[1]])
    RF_H_z=np.array([A.RF_S_pos[2],A.RF_K_pos[2]])
    ax.plot3D(RF_H_x, RF_H_y, RF_H_z, 'red')
    RF_F_x=np.array([A.RF_K_pos[0],A.RF_F_pos[0]])
    RF_F_y=np.array([A.RF_K_pos[1],A.RF_F_pos[1]])
    RF_F_z=np.array([A.RF_K_pos[2],A.RF_F_pos[2]])
    ax.plot3D(RF_F_x, RF_F_y, RF_F_z, 'blue')
    #LR
    LR_S_x=np.array([-A.B_L/2,A.LR_S_pos[0]])
    LR_S_y=np.array([A.B_W/2,A.LR_S_pos[1]])
    LR_S_z=np.array([0,A.LR_S_pos[2]])
    ax.plot3D(LR_S_x, LR_S_y, LR_S_z, 'green')
    LR_H_x=np.array([A.LR_S_pos[0],A.LR_K_pos[0]])
    LR_H_y=np.array([A.LR_S_pos[1],A.LR_K_pos[1]])
    LR_H_z=np.array([A.LR_S_pos[2],A.LR_K_pos[2]])
    ax.plot3D(LR_H_x, LR_H_y, LR_H_z, 'red')
    LR_F_x=np.array([A.LR_K_pos[0],A.LR_F_pos[0]])
    LR_F_y=np.array([A.LR_K_pos[1],A.LR_F_pos[1]])
    LR_F_z=np.array([A.LR_K_pos[2],A.LR_F_pos[2]])
    ax.plot3D(LR_F_x, LR_F_y, LR_F_z, 'blue')
    #RR
    RR_S_x=np.array([-A.B_L/2, A.RR_S_pos[0]])
    RR_S_y=np.array([-A.B_W/2, A.RR_S_pos[1]])
    RR_S_z=np.array([0,A.RR_S_pos[2]])
    ax.plot3D(RR_S_x, RR_S_y, RR_S_z, 'green')
    RR_H_x=np.array([A.RR_S_pos[0],A.RR_K_pos[0]])
    RR_H_y=np.array([A.RR_S_pos[1],A.RR_K_pos[1]])
    RR_H_z=np.array([A.RR_S_pos[2],A.RR_K_pos[2]])
    ax.plot3D(RR_H_x, RR_H_y, RR_H_z, 'red')
    RR_F_x=np.array([A.RR_K_pos[0],A.RR_F_pos[0]])
    RR_F_y=np.array([A.RR_K_pos[1],A.RR_F_pos[1]])
    RR_F_z=np.array([A.RR_K_pos[2],A.RR_F_pos[2]])
    ax.plot3D(RR_F_x, RR_F_y, RR_F_z, 'blue')
    if(with_box):#plotting the legs workspace
        #LF
        for i in range(0,4):
            X=np.array([A.Lower_bnd[i,0], A.Lower_bnd[i,0], A.Upper_bnd[i,0], A.Upper_bnd[i,0], A.Lower_bnd[i,0]])
            Y=np.array([A.Lower_bnd[i,1], A.Upper_bnd[i,1], A.Upper_bnd[i,1], A.Lower_bnd[i,1], A.Lower_bnd[i,1]])
            Z=np.array([A.Lower_bnd[i,2], A.Lower_bnd[i,2], A.Lower_bnd[i,2], A.Lower_bnd[i,2], A.Lower_bnd[i,2]])
            ax.plot3D(X, Y, Z, '--c')
            Z=np.array([A.Upper_bnd[i,2], A.Upper_bnd[i,2], A.Upper_bnd[i,2], A.Upper_bnd[i,2], A.Upper_bnd[i,2]])
            ax.plot3D(X, Y, Z, '--c')
            X=np.array([A.Lower_bnd[i,0], A.Lower_bnd[i,0], A.Lower_bnd[i,0], A.Lower_bnd[i,0], A.Lower_bnd[i,0]])
            Y=np.array([A.Lower_bnd[i,1], A.Upper_bnd[i,1], A.Upper_bnd[i,1], A.Lower_bnd[i,1], A.Lower_bnd[i,1]])
            Z=np.array([A.Lower_bnd[i,2], A.Lower_bnd[i,2], A.Upper_bnd[i,2], A.Upper_bnd[i,2], A.Lower_bnd[i,2]])
            ax.plot3D(X, Y, Z, '--c')
            X=np.array([A.Upper_bnd[i,0], A.Upper_bnd[i,0], A.Upper_bnd[i,0], A.Upper_bnd[i,0], A.Upper_bnd[i,0]])
            ax.plot3D(X, Y, Z, '--c')
    
    # ax.set_aspect('equal', adjustable='box')
    ax.view_init( 45, -90)


# Initializing the spot class
A=Spot(0.13,0.105,0.05,0.08,0.222)
D=np.linalg.norm(A.Goal-A.pos)
P=A.F_pos
#while D>A.Walk_D:
plot_spot(A,0)
DX=np.array([A.Walk_D,0.,0.])
A.LF_Inv_kin(-DX/3)
A.RF_Inv_kin(DX/3)
A.LR_Inv_kin(2*DX/3)
plot_spot(A,0)

# for i in range(0,10):
#     A.Tragectory_controller()
#     #plot_spot(A,0)
    
#     #print(A.Goal-A.pos)
#     print(A.Current_swing)
  

# plot_spot(A,1)
# DX=np.array([0.1,0.,0.])
# A.LF_Inv_kin(DX)
# plot_spot(A,1)
# A.RF_Inv_kin(DX)
# plot_spot(A,1)
# A.LR_Inv_kin(DX)
# plot_spot(A,1)
# A.RR_Inv_kin(DX)
# plot_spot(A,1)

