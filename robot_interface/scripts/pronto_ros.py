#!/usr/bin/env python3

import rospy
import numpy as np
from numpy import linalg as LA, ndarray
import math
from pyquaternion import Quaternion
import random
import matplotlib.pyplot as plt
import timeit
from unitree_legged_msgs.msg import A1LowState

f = 100  # Hz
dt = 1 / f
I_3 = np.eye(3, dtype=float)

# State model noise covariance matrix Q_k
Q = np.random.normal(0, 1, size=(6, 6)) * 0.001

# Measurement matrix H_k
H = np.zeros((3, 15))
H[:, 6:9] = I_3  # V_mes

# FOR FIRST ITERATION
state_estimate_k_minus = np.zeros((15, 1))
covariance_estimate_k_minus = np.random.normal(0, 1, size=(15, 15)) * 0.001

# a1 const
"""
    Leg0 FR=right front leg
    Leg1 FL=left front leg
    Leg2 RR=right rear leg
    Leg3 RL=left rear leg
    Joint0=Hip, Hip joint
    Joint1=Thigh, Thigh joint
    Joint2=Calf, Calf joint"""

FR = 0  # leg index
FL = 1
RR = 2
RL = 3

FR_0 = 0  # joint index
FR_1 = 1
FR_2 = 2

FL_0 = 3
FL_1 = 4
FL_2 = 5

RR_0 = 6
RR_1 = 7
RR_2 = 8

RL_0 = 9
RL_1 = 10
RL_2 = 11

Leg_offset_x = 0.1805  # robot x length(length) 0.1805*2
Leg_offset_y = 0.047  # robot y length(width) 0.047*2
Trunk_offset_z = 0.01675  # robot z length(height)
Shoulder_width = 0.0838  # from Hip servo to Thigh servo
Upper_leg_length = 0.2  # from Thigh to Calf
Lower_leg_length = 0.2  # from Calf to Foot
Foot_radius = 0.2
A1_mass = 11  # kg

lay_ang = np.array([0.0, 30 * math.pi / 180, -60 * math.pi / 180])  #### guess!####
stand_ang = np.array([0.0, 30 * math.pi / 180, -60 * math.pi / 180])  ####guess!####


# transformation matrix from foot to center of the shoulder(hips)
def forward_kinematics(angle):
    px_0 = np.zeros(4)
    py_0 = np.zeros(4)
    pz_0 = np.zeros(4)
    l1 = Shoulder_width
    l2 = Upper_leg_length
    l3 = Lower_leg_length
    for i in range(4):
        h = angle[i, 0]  # teta1 of leg i
        t = angle[i, 1]  # teta2 of leg i
        c = angle[i, 2]  # teta3 of leg i
        # 0 is relative to shoulder
        px_0[i] = math.cos(h) * (l1 + math.cos(t) * l2 + math.cos(t + c) * l3)
        py_0[i] = math.sin(h) * (l1 + math.cos(t) * l2 + math.cos(t + c) * l3)
        pz_0[i] = math.sin(t) * l2 + math.sin(t + c) * l3
        """
        T_foot_shoulder = np.array([[ math.cos(h)*math.cos(t+c),-math.cos(h)*math.sin(t+c),math.sin(h),px_0],
                    [math.sin(h)*math.cos(t+c),-math.sin(h)*math.sin(t+c),-math.cos(h),py_0],
                    [math.sin(t+c),math.cos(t+c),0,pz_0],
                    [0.0,0.0,0.0,1.0]])
                    transformation matrix
                    """
    pos_in_shoulder: ndarray = np.array([px_0, py_0, pz_0])
    pos_in_shoulder.transpose()
    return pos_in_shoulder


# transformation matrix from shoulder to base
def shoulder_2_base(pos_in_shoulder):
    pos_in_base = np.zeros((4, 3))
    for foot_num in range(4):
        if foot_num == 0 or foot_num == 1:
            alpha = 1
        else:
            alpha = -1
        if foot_num == 1 or foot_num == 3:
            beta = 1
        else:
            beta = -1

        T_shoulder_base = np.array([[0.0, 0.0, 1.0, alpha * Leg_offset_x / 2],
                                    [0.0, 1.0, 0.0, beta * Leg_offset_y / 2],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
        foot_pos = pos_in_shoulder[:, foot_num]
        foot_pos = np.append(foot_pos, [1.0])
        pos_in_base[foot_num, :] = (T_shoulder_base @ foot_pos)[0:3]
    return pos_in_base


def fk_Jacobian(leg_ang):  # Jacobian of forward kinematics
    h = leg_ang[0]  # teta1
    t = leg_ang[1]  # teta2
    c = leg_ang[2]  # teta3
    l1 = Shoulder_width
    l2 = Upper_leg_length
    l3 = Lower_leg_length

    #  0 is relative to shoulder
    Jacobian = np.array([[0.0, l2 * math.cos(t) + l3 * math.cos(t + c), l3 * math.cos(t + c)],
                         [math.cos(h) * (l1 + l2 * math.cos(t) + l3 * math.cos(t + c)),
                          -math.sin(h) * (l2 * math.sin(t) + l3 * math.sin(t + c)),
                          -l3 * math.sin(h) * math.sin(t + c)],
                         [-math.sin(h) * (l1 + l2 * math.cos(t) + l3 * math.cos(t + c)),
                          math.sin(h) * (l2 * math.sin(t) + l3 * math.sin(t + c)), l3 * math.cos(h) * math.sin(t + c)]])
    return Jacobian


def ekf(z_k_observation_vector, state_estimate_k_, P_k_v, A_k, W_k):
    global covariance_estimate_k_minus, state_estimate_k_minus
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.
    OUTPUT
        return state_estimate_k near-optimal state estimate at time k
    """

    # Predict #
    # Predict the state estimate at time k based on the state
    # estimate at time k-1 and the control input applied at time k-1.

    # print("State Estimate Before EKF=", state_estimate_k)

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    covariance_estimate_k_ = A_k @ covariance_estimate_k_minus.T @ A_k.T + W_k @ Q @ W_k.T

    # Update (Correct) #
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted
    # the sensor measurements would be for the current time step k.
    measurement_residual_y_k = z_k_observation_vector.T - (H @ state_estimate_k_)

    # print("Observation=", z_k_observation_vector)

    # Calculate the measurement residual covariance
    S_k = H @ covariance_estimate_k_ @ H.T + P_k_v

    # Calculate the near-optimal Kalman gain
    # We use pseudocode since some matrices might be
    # non-square or singular.
    K_k = covariance_estimate_k_ @ H.T @ np.linalg.pinv(S_k)

    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k_ + (K_k @ measurement_residual_y_k)
    state_estimate_k_minus = state_estimate_k

    # Update the state covariance estimate for time k
    covariance_estimate_k = covariance_estimate_k_ - (K_k @ H @ covariance_estimate_k_)
    covariance_estimate_k_minus = covariance_estimate_k
    # Print the best (near-optimal) estimate of the current state of the robot
    # print("State Estimate After EKF=", state_estimate_k)

    # Return the updated state and covariance estimates
    return state_estimate_k, covariance_estimate_k


def skew_symmetric_matrix(vector):
    vector = vector[0]
    # this function returns the skew symmetric cross product matrix for vector.
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    w, x, y, z = quaternion.elements
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return np.array([[roll_x, pitch_y, yaw_z]])  # in radians


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(qx, qy, qz, qw)


def pronto(LowState):
    global state_estimate_k_minus, covariance_estimate_k_minus
    # const

    b_q = Quaternion()  # q noise
    collect_v_noise = np.array([[0.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, -9.81]])
    servo_Angle_mes = np.zeros((4, 3))
    angular_velocity_mes = np.zeros((4, 3))
    v_from_leg_k = np.zeros((4, 3))
    v_leg_odometry = np.array([0.0, 0.0, 0.0])

    # read last step

    r_minus = state_estimate_k_minus[0:3].T
    euler_rotation = state_estimate_k_minus[3:6]
    roll, pitch, yaw = euler_rotation
    q_k_minus = get_quaternion_from_euler(roll, pitch, yaw)
    v_minus = state_estimate_k_minus[6:9].T
    b_a_minus = state_estimate_k_minus[9:12].T  # acc noise
    b_w_minus = state_estimate_k_minus[12:15].T  # omega noise

    state_estimate_k_ = state_estimate_k_minus
    # read low state massage
    # Measurement
    imu_acc = np.array(LowState.accelerometer)
    print('imu_acc', imu_acc)
    imu_omega = np.array(LowState.gyroscope)
    print('imu_omega', imu_omega)
    # q_mes = np.array(LowState.quaternion)
    # imu_quaternion = Quaternion(q_mes)

    Hip_angle = np.array([LowState.q[FR_0], LowState.q[FL_0], LowState.q[RR_0],
                          LowState.q[RL_0]])
    Thigh_angle = np.array([LowState.q[FR_1], LowState.q[FL_1], LowState.q[RR_1],
                            LowState.q[RL_1]])
    Calf_angle = np.array([LowState.q[FR_2], LowState.q[FL_2], LowState.q[RR_2],
                           LowState.q[RL_2]])

    Hip_Angular_velocity = np.array(
        [LowState.dq[FR_0], LowState.dq[FL_0], LowState.dq[RR_0],
         LowState.dq[RL_0]])
    Thigh_Angular_velocity = np.array(
        [LowState.dq[FR_1], LowState.dq[FL_1], LowState.dq[RR_1],
         LowState.dq[RL_1]])
    Calf_Angular_velocity = np.array(
        [LowState.dq[FR_2], LowState.dq[FL_2], LowState.dq[RR_2],
         LowState.dq[RL_2]])

    for i in range(4):  # 4 legs
        for j in range(3):  # 3 arms
            servo_Angle_mes[i, :] = ([Hip_angle[i], Thigh_angle[i], Calf_angle[i]])
            angular_velocity_mes[i, :] = (
                [Hip_Angular_velocity[i], Thigh_Angular_velocity[i], Calf_Angular_velocity[i]])

    isStep = np.array(LowState.footForce)  # which leg is stepping
    num_of_step_leg = sum(isStep)  # number of leg that stepping

    # Update state
    Pos_in_shoulder = forward_kinematics(servo_Angle_mes)
    Pos_in_base = shoulder_2_base(Pos_in_shoulder)
    # print("Pos_in_base:", Pos_in_base)

    acc_body = imu_acc - b_a_minus  # In body frame
    omega = imu_omega - b_w_minus

    rotation_angle = dt * LA.norm(omega)
    print(rotation_angle)
    rotation_axis = omega / LA.norm(omega)
    print(rotation_axis)
    # quaternion update
    # Rotate in rotation_angle about rotation_axis, when rotation_angle is in radians
    q_update = Quaternion(axis=rotation_axis[0], angle=rotation_angle)
    print(q_update)
    q_k = q_update * q_k_minus
    print(q_k)
    print(acc_body)
    acc_world = q_k.rotate(acc_body[0])
    print('LowState.accelerometer:', LowState.accelerometer)
    print('b_a_minus:', b_a_minus)
    print('acc_body:', acc_body)
    print('acc_world:', acc_world)
    rotation_matrix = q_k.rotation_matrix  # quaternion to rotation_matrix of base to world
    euler_rotation = euler_from_quaternion(q_k)
    v = v_minus + dt * (acc_world + g)
    r = r_minus + dt * v_minus + math.pow(dt, 2) / 2 * (acc_world + g)
    print('r=', r)
    b_a = b_a_minus
    b_w = b_w_minus

    # The estimated state vector at time k-1
    # x=[p,R,v,b_a,b_w]
    f_c = np.concatenate((r, euler_rotation, v, b_a, b_w), axis=1).T
    v_skew = skew_symmetric_matrix(v)
    omega_skew = skew_symmetric_matrix(omega)

    # Calculate A_c matrix
    zero_mat = np.zeros((3, 3))
    d_p_dot_dx = np.concatenate((zero_mat, -rotation_matrix * v_skew, rotation_matrix, zero_mat, zero_mat), axis=1)
    d_rotation_matrix_dot_dx = np.concatenate((zero_mat, - omega_skew, zero_mat, zero_mat, zero_mat), axis=1)
    d_v_dot_dx = np.concatenate((zero_mat, (skew_symmetric_matrix(rotation_matrix.T * g)), -omega_skew, zero_mat,
                                 zero_mat), axis=1)
    d_b_a_dot_dx = np.zeros((3, 15))
    d_b_w_dot_dx = np.zeros((3, 15))
    A_c = np.concatenate((d_p_dot_dx, d_rotation_matrix_dot_dx, d_v_dot_dx, d_b_a_dot_dx, d_b_w_dot_dx), axis=0)

    # Calculate W_c matrix
    d_p_dot_du = np.concatenate((zero_mat, zero_mat), axis=1)
    d_rotation_matrix_dot_du = np.concatenate((zero_mat, I_3), axis=1)
    d_v_dot_du = np.concatenate((I_3, v_skew), axis=1)
    d_b_a_dot_du = np.zeros((3, 6))
    d_b_w_dot_du = np.zeros((3, 6))
    W_c = np.concatenate((d_p_dot_du, d_rotation_matrix_dot_du, d_v_dot_du, d_b_a_dot_du, d_b_w_dot_du), axis=0)

    # Calculate state
    state_estimate_k_ = state_estimate_k_ + f_c * dt
    A_k = np.eye(15, dtype=float) + A_c * dt
    W_k = W_c * dt

    # EKF
    # Create a list of sensor observations
    # form pronto
    for i in range(4):  # 4 legs
        J = fk_Jacobian(servo_Angle_mes[i])
        v_from_leg_k[i] = -J @ angular_velocity_mes[i, :] - np.cross(omega, Pos_in_base[i, :]) + collect_v_noise
        v_leg_odometry = v_leg_odometry + v_from_leg_k[i, :] * (1 * isStep[i])
    if num_of_step_leg:  # check that num_of_step_leg is not 0
        v_leg_odometry = v_leg_odometry / num_of_step_leg + collect_v_noise
        z_k = v_leg_odometry  # from leg odematry
    else:
        v_leg_odometry = np.zeros((1, 4))
        z_k = v

    # covariance for the velocity measurement P_k calculate
    delta_v = v_leg_odometry - v_from_leg_k[isStep == 1, :]
    D_k = 1 / num_of_step_leg * (delta_v.T @ delta_v)
    delta_force = 0
    alpha = 1  # guess
    P_0_v = np.zeros((3, 3))  # my guess
    P_k_v = P_0_v + np.power((0.5 * D_k + I_3 * delta_force / alpha), 2)

    #  Run the Extended Kalman Filter
    optimal_state_estimate_k, covariance_estimate_k = ekf(
        z_k,  # Most recent sensor measurement
        state_estimate_k_,  # Our most recent estimate of the state
        P_k_v, A_k, W_k)  # Our most recent state covariance matrix)

    # # Print a blank line
    # print()
    # print()
    # print("imu_quaternion", imu_quaternion)
    # print("omega", omega)
    # print("imu_acc:", acc_body)
    #
    # print("Hip_angle:", Hip_angle)
    # print("Thigh_angle:", Thigh_angle)
    # print("Calf_angle:", Calf_angle)
    #
    # print("FR_pos_in_base", Pos_in_base[0, :])
    # print("FL_pos_in_base", Pos_in_base[1, :])
    # print("RR_pos_in_base", Pos_in_base[2, :])
    # print("RL_pos_in_base", Pos_in_base[3, :])
    #
    # print("FR_footForce", Foot_Force_mes[0])
    # print("FL_footForce", Foot_Force_mes[1])
    # print("RR_footForce", Foot_Force_mes[2])
    # print("RL_footForce", Foot_Force_mes[3])
    #
    # print("C:", rotation_matrix)
    # print("acc_world:", acc_world)
    # print("q_k or q_DOT:", q_k)
    #
    # print("r:", r)
    # print("v:", v)
    # print("v_mes:", v_leg_odometry)
    return optimal_state_estimate_k, covariance_estimate_k


def pronto_mes(A1LowState):

    optimal_state_estimate_k, covariance_estimate_k = pronto(A1LowState)
    print(optimal_state_estimate_k)

def listener():
    rospy.init_node('pronto', anonymous=True)

    rospy.Subscriber("low_state_Lo_res", A1LowState, pronto_mes)

    rospy.spin()


if __name__ == '__main__':
    listener()
