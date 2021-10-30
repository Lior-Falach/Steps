"This file defines the relevent geometric models Forward & inverse Kinematics and Dynamics "
import numpy as np

# in the following
import numpy as np
from numpy import linalg as LA
import math

class Quaternion:

    def creat_quaternion(self,w,vector):#creat quaternion
        x, y, z = vector
        return np.array([w,x,y,z])

    def rotation_in_quaternion(self,angle,rotation_axis):
        #A rotation about the unit vector n^ by an angle, computed using the quaternion 

        rotation_axis=LA.norm(rotation_axis) #in case that rotation_axis is not in unit size
        qw=math.cos(0.5*angle)
        q =math.sin(0.5*angle)*rotation_axis
        qx=q[0]
        qy=q[1]
        qz=q[2]
        return np.array([qx,qy,qz,qw])

    def quaternion_conjugate(self,quaternion):#q*
        w, x, y, z = quaternion
        return np.array([w,-x,-y,-z])

    def quaternion_inverse(self,quaternion):#q^-1
        return self.quaternion_conjugate(quaternion)/(math.pow(LA.norm(quaternion),2))

    def quaternion_multiply(self,quaternion1, quaternion0):#q1*q2
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1

        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def quaternion_2_rotation_matrix(self,quaternion):
        #q (1x4) -> R (3x3)

        # Extract the values from quaternion
        q0, q1, q2, q3 = quaternion
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix

    def quaternion_update(self,old_quaternion,rotation_in_quaternion):
        #q(t+1)=q(t)*rotation_in_quaternion
        return self.quaternion_multiply(old_quaternion,rotation_in_quaternion)

    def vec_rotation_update(self,vector,rotation_quaternion):
        #return rotation vector 
        vec_in_quaternion=self.creat_quaternion(0,np.array([vector[0],vector[1],vector[2]]))
        rotation_quaternion_inverse=self.quaternion_inverse(rotation_quaternion)
        rotation_vec_in_quaternion=self.quaternion_multiply(self.quaternion_multiply(rotation_quaternion,vec_in_quaternion),rotation_quaternion_inverse)
        w, vec_new_x, vec_new_y, vec_new_z = rotation_vec_in_quaternion
        return np.array[vec_new_x, vec_new_y, vec_new_z]
