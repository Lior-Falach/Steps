import numpy as np
from numpy import linalg as LA
import math
from pyquaternion import Quaternion
import random
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from numpy import diff

from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.fft import fft, fftfreq

f = 473
dt = 1 / f
I_3 = np.eye(3, dtype=float)

# State model noise covariance matrix Q_k
deg2rad = math.pi/180
Q_acc_xx = 0.01
Q_acc_xy = 0.0
Q_acc_xz = 0.0
Q_acc_yy = 0.0
Q_acc_yz = 0.0
Q_acc_zz = 0.0
Q_omega_xx = 0.1*deg2rad
Q_omega_xy = 0.1*deg2rad
Q_omega_xz = 0.1*deg2rad
Q_omega_yy = 0.1*deg2rad
Q_omega_yz = 0.1*deg2rad
Q_omega_zz = 0.1*deg2rad
Q = np.array([[Q_acc_xx, Q_acc_xy, Q_acc_xz, 0.0, 0.0, 0.0],
              [Q_acc_xy, Q_acc_yy, Q_acc_yz, 0.0, 0.0, 0.0],
              [Q_acc_xz, Q_acc_yz, Q_acc_zz, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, Q_omega_xx, Q_omega_xy, Q_omega_xz],
              [0.0, 0.0, 0.0, Q_omega_xy, Q_omega_yy, Q_omega_yz],
              [0.0, 0.0, 0.0, Q_omega_xz, Q_omega_yz, Q_omega_zz]])
# Q = np.array([[1.5, -1.44927732, 0.95105996, -0.45190575, -1.69205024, 1.49538745],
#               [1.57751834, -1.01320238, -1.97360812, -1.48444642, 0.50197512, 1.3540551],
#               [-0.847046, -0.02991231, -1.70271629, 0.37877357, -1.62739234, 0.67124554],
#               [1.61041854, -0.03259412, -2.66872417, -0.60406391, -0.43124619, -0.60861009],
#               [-0.21390035, 1.47023247, -1.81129592, 1.16655097, 0.98225882, 0.32267955],
#               [-1.57084544, -0.14073233, 2.37634776, -1.86233095, -0.28480153, 0.23114744]])  #from mc

# Measurement matrix H_k
H = np.zeros((3, 15))
H[:, 6:9] = I_3
P_0_v = np.array([1.0, 1.0, 1.0])*0.00001

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
Trunk_offset_z = 0.26  # robot z length(height)
Shoulder_width = 0.0838  # from Hip servo to Thigh servo
Upper_leg_length = 0.2  # from Thigh to Calf
Lower_leg_length = 0.2  # from Calf to Foot
Foot_radius = 0.2
A1_mass = 11  # kg

collect_v_noise = np.random.normal(0, 1, size=(1, 3)) * 0 / 1000  # mes_velocity_noise
acc_noise = np.random.normal(0, 1, size=(1, 3)) * 0 / 1000  # mes_acc_noise
omega_noise = np.random.normal(0, 1, size=(1, 3)) * 0 / 1000  # mes_omega_noise
g = np.array([[0.0, 0.0, 0.0]])  # acc mes don't include gravity
servo_Angle_mes = np.zeros((4, 3))
angular_velocity_mes = np.zeros((4, 3))
v_from_leg_k = np.zeros((4, 3))


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
    pos_in_shoulder = np.array([px_0, py_0, pz_0])
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
        pos_in_base[foot_num, :] = np.matmul(T_shoulder_base, foot_pos)[0:3]
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


def ekf(z_k_observation_vector, state_estimate_k_,
        covariance_estimate_k_minus, P_k_v, A_k, W_k):
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

    # Update the state covariance estimate for time k
    covariance_estimate_k = covariance_estimate_k_ - (K_k @ H @ covariance_estimate_k_)
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

    return Quaternion(qw, qx, qy, qz)


def read_sensor(LowState, state_estimate_k_minus, covariance_estimate_k_minus, reset_commend, Last_filter_acc, Last_filter_omega):
    v_leg_odometry = np.array([[0.0, 0.0, 0.0]])

    # read last step
    r_minus = state_estimate_k_minus[0:3].T
    euler_rotation = state_estimate_k_minus[3:6]
    roll, pitch, yaw = euler_rotation
    q_k_minus = get_quaternion_from_euler(roll, pitch, yaw)
    rotation_matrix_minus = q_k_minus.rotation_matrix
    v_minus = state_estimate_k_minus[6:9].T
    b_a_minus = state_estimate_k_minus[9:12].T  # acc noise
    b_w_minus = state_estimate_k_minus[12:15].T  # omega noise
    # read low state massage
    # Measurement
    # Low Pass Filter On Mes
    Cutoff_fre = 5  # Hz
    Cutoff_angular_velocity = 2 * math.pi * Cutoff_fre
    Sampling_period = dt
    alpha = Cutoff_angular_velocity * Sampling_period / (1 + Cutoff_angular_velocity * Sampling_period)
    alpha = 1
    imu_acc = np.array(alpha * LowState.imu.accelerometer + (1 - alpha) * Last_filter_acc)
    Cutoff_fre = 50  # Hz
    Cutoff_angular_velocity = 2 * math.pi * Cutoff_fre
    Sampling_period = dt
    alpha = Cutoff_angular_velocity * Sampling_period / (1 + Cutoff_angular_velocity * Sampling_period)
    alpha = 1
    imu_omega = alpha * LowState.imu.gyroscope + (1 - alpha) * Last_filter_omega

    Hip_angle = np.array([LowState.motorState[FR_0].q, LowState.motorState[FL_0].q, LowState.motorState[RR_0].q,
                          LowState.motorState[RL_0].q])
    Thigh_angle = np.array([LowState.motorState[FR_1].q, LowState.motorState[FL_1].q, LowState.motorState[RR_1].q,
                            LowState.motorState[RL_1].q])
    Calf_angle = np.array([LowState.motorState[FR_2].q, LowState.motorState[FL_2].q, LowState.motorState[RR_2].q,
                           LowState.motorState[RL_2].q])

    Hip_Angular_velocity = np.array(
        [LowState.motorState[FR_0].dq, LowState.motorState[FL_0].dq, LowState.motorState[RR_0].dq,
         LowState.motorState[RL_0].dq])
    Thigh_Angular_velocity = np.array(
        [LowState.motorState[FR_1].dq, LowState.motorState[FL_1].dq, LowState.motorState[RR_1].dq,
         LowState.motorState[RL_1].dq])
    Calf_Angular_velocity = np.array(
        [LowState.motorState[FR_2].dq, LowState.motorState[FL_2].dq, LowState.motorState[RR_2].dq,
         LowState.motorState[RL_2].dq])

    for leg in range(4):  # 4 legs
        servo_Angle_mes[leg, :] = ([Hip_angle[leg], Thigh_angle[leg], Calf_angle[leg]])
        angular_velocity_mes[leg, :] = (
            [Hip_Angular_velocity[leg], Thigh_Angular_velocity[leg], Calf_Angular_velocity[leg]])

    isStep = np.array(LowState.footForce)  # which leg is stepping
    num_of_step_leg = sum(isStep)  # number of leg that stepping

    # Update leg pos
    Pos_in_shoulder = forward_kinematics(servo_Angle_mes)
    Pos_in_base = shoulder_2_base(Pos_in_shoulder)

    # check if the robot stand more than 200 msec
    if reset_commend:
        b_a_minus = imu_acc
        b_w_minus = imu_omega
        acc_body = imu_acc - b_a_minus - acc_noise  # In body frame
        omega = imu_omega - b_w_minus - omega_noise
        q_k = q_k_minus
    else:
        acc_body = imu_acc - b_a_minus - acc_noise  # In body frame
        omega = imu_omega - b_w_minus - omega_noise
        # quaternion update
        # Rotate in rotation_angle about rotation_axis, when rotation_angle is in radians
        rotation_angle = dt * LA.norm(omega)
        rotation_axis = omega / LA.norm(omega)
        q_update = Quaternion(axis=rotation_axis[0], angle=rotation_angle)
        q_k = q_update * q_k_minus

    rotation_matrix = q_k.rotation_matrix  # quaternion to rotation_matrix of base to world
    euler_rotation = euler_from_quaternion(q_k)
    # update state
    # x=[p,R,v,b_a,b_w]
    v = v_minus + dt * (
            (- skew_symmetric_matrix(omega) @ v_minus.T).T + (rotation_matrix_minus.T @ g.T).T + acc_body)
    v = v_minus + dt * acc_body
    r = r_minus + (rotation_matrix_minus @ (dt * v_minus).T).T
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
    state_estimate_k_ = f_c
    A_k = np.eye(15, dtype=float) + A_c * dt
    W_k = W_c * dt

    # EKF
    # Create a list of sensor observations
    # form pronto
    for leg in range(4):  # 4 legs
        J = fk_Jacobian(servo_Angle_mes[leg])
        v_from_leg_k[leg] = -J @ angular_velocity_mes[leg, :] - np.cross(omega, Pos_in_base[leg, :]) - collect_v_noise
        v_leg_odometry = v_leg_odometry + v_from_leg_k[leg, :] * (1 * isStep[leg])
    if reset_commend:  # check that num_of_step_leg is not 0
        v_leg_odometry = np.zeros((1, 3))
        delta_v = np.zeros((4, 3))
    else:  # the robot is not touch the ground
        v_leg_odometry = v_leg_odometry / num_of_step_leg - collect_v_noise
        delta_v = v_leg_odometry - v_from_leg_k[isStep == 1, :]
    r_leg_odometry = r_minus + v_leg_odometry * dt
    z_k = v_leg_odometry
    if num_of_step_leg > 2:
        v_leg_odometry = v
        z_k = v

    # covariance for the velocity measurement P_k calculate
    D_k = 1 / num_of_step_leg * (delta_v.T @ delta_v)
    delta_force = np.random.rand(1, 1)
    alpha = 5  # guess
    P_k_v = P_0_v + np.power((0.5 * D_k + I_3 * delta_force / alpha), 2)

    #  Run the Extended Kalman Filter
    optimal_state_estimate_k, covariance_estimate_k = ekf(
        z_k,  # Most recent sensor measurement
        state_estimate_k_,  # Our most recent estimate of the state
        covariance_estimate_k_minus,  # Our most recent estimate of the cov
        P_k_v, A_k, W_k)  # Our most recent state covariance matrix)
    return optimal_state_estimate_k, covariance_estimate_k, Pos_in_base, isStep, rotation_matrix, imu_acc, acc_body, z_k, state_estimate_k_, imu_omega, r_leg_odometry


class Imu:
    def __init__(self, accelerometer, gyroscope, quaternion):
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.quaternion = quaternion

    def __repr__(self):
        return "accelerometer:% s gyroscope:% s quaternion:% s" % (self.accelerometer, self.gyroscope,
                                                                   self.quaternion)


class MotorState:
    def __init__(self, q, dq):
        self.q = q
        self.dq = dq

    def __repr__(self):
        return "q:% s dq:% s" % (self.q, self.dq)


class LowState:
    def __init__(self, q, dq, accelerometer, gyroscope, quaternion, footForce):
        self.imu = Imu(accelerometer, gyroscope, quaternion)
        motorState = []
        for i in range(12):
            motorState.append(MotorState(q[i], dq[i]))
        self.motorState = motorState
        self.footForce = footForce

    def __repr__(self):
        return "imu:% s motorState:% s footForce:% s" % (self.imu, self.motorState, self.footForce)


if __name__ == '__main__':
    time = 0
    t = []
    x = []
    y = []
    z = []
    x_imu = []
    y_imu = []
    z_imu = []
    x_mes = []
    y_mes = []
    z_mes = []
    vx = []
    vy = []
    vz = []
    vx_imu = []
    vy_imu = []
    vz_imu = []
    vx_mes = []
    vy_mes = []
    vz_mes = []
    filter_acc_x = []
    filter_acc_y = []
    filter_acc_z = []
    acc_after_bias_x = []
    acc_after_bias_y = []
    acc_after_bias_z = []
    acc_x = []
    acc_y = []
    acc_z = []
    R = []
    Pos_in_base = []
    isStep = []
    # read data
    temp_data = pd.read_csv('/home/tal/catkin_ws/src/Steps/robot_interface/sim_result_new.txt')
    data = temp_data.values
    temp_real_data = pd.read_csv('/home/tal/catkin_ws/src/Steps/robot_interface/sim_result_real_new.txt')
    real_data = temp_real_data.values

    # FOR FIRST ITERATION
    reset_commend = False
    delta_time_stepping = 0
    # FOR FIRST ITERATION
    # lowState_k_minus = LowState(data[1][1:13], data[1][13:25], data[1][60:63], data[1][57:60], data[1][53:57],
    #                             data[1][49:53])
    imu_acc_k_minus = np.array([data[1][60:63]])
    imu_omega_k_minus = np.array([data[1][57:60]])
    r_0 = np.array([real_data[1, 1:4]])
    euler_rotation_0 = euler_from_quaternion(Quaternion(real_data[1, 4:8]))
    v_0 = np.array([[0.0, 0.0, 0.0]])
    b_a_0 = np.array([[0.0, 0.0, 0.0]])
    b_w_0 = np.array([[0.0, 0.0, 0.0]])

    state_estimate_k_minus = np.concatenate((r_0, euler_rotation_0, v_0, b_a_0, b_w_0), axis=1).T
    covariance_estimate_k_minus = np.zeros((15, 15)) + 0.0001
    i = 1
    number_of_steps = len(real_data)
    RUN_TIME = 20
    while time < RUN_TIME:  # i < number_of_steps-1:  # time < RUN_TIME:
        q = data[i][1:13]
        dq = data[i][13:25]
        footForce = data[i][49:53]
        quaternion = data[i][53:57]
        gyroscope = data[i][57:60]
        accelerometer = data[i][60:63]
        i += 1
        lowState_k = LowState(q, dq, accelerometer, gyroscope, quaternion, footForce)  # read sensors
        optimal_state_estimate_k, covariance_estimate_k, leg_pos, is_step, rotation_matrix, filter_acc, acc_after_bias, z_k, state_estimate_no_ekf, filter_omega, r_leg_odometry\
            = read_sensor(
            lowState_k,
            state_estimate_k_minus,
            covariance_estimate_k_minus,
            reset_commend,
            imu_acc_k_minus, imu_omega_k_minus)
        t.append(time)
        x.append(optimal_state_estimate_k[0][0])
        y.append(optimal_state_estimate_k[1][0])
        z.append(optimal_state_estimate_k[2][0])
        x_imu.append(state_estimate_no_ekf[0][0])
        y_imu.append(state_estimate_no_ekf[1][0])
        z_imu.append(state_estimate_no_ekf[2][0])
        x_mes.append(r_leg_odometry[0][0])
        y_mes.append(r_leg_odometry[0][1])
        z_mes.append(r_leg_odometry[0][2])
        vx.append(optimal_state_estimate_k[6][0])
        vy.append(optimal_state_estimate_k[7][0])
        vz.append(optimal_state_estimate_k[8][0])
        vx_imu.append(state_estimate_no_ekf[6][0])
        vy_imu.append(state_estimate_no_ekf[7][0])
        vz_imu.append(state_estimate_no_ekf[8][0])
        vx_mes.append(z_k[0][0])
        vy_mes.append(z_k[0][1])
        vz_mes.append(z_k[0][2])
        filter_acc_x.append(filter_acc[0][0])
        filter_acc_y.append(filter_acc[0][1])
        filter_acc_z.append(filter_acc[0][2])
        acc_after_bias_x.append(acc_after_bias[0][0])
        acc_after_bias_y.append(acc_after_bias[0][1])
        acc_after_bias_z.append(acc_after_bias[0][2])
        acc_x.append(accelerometer[0])
        acc_y.append(accelerometer[1])
        acc_z.append(accelerometer[2])
        Pos_in_base.append(leg_pos)
        isStep.append(is_step)
        R.append(rotation_matrix)
        print()
        print()
        print("time:", time)
        # print('optimal_state_estimate_k: ', optimal_state_estimate_k)
        # print('covariance_estimate_k: ', covariance_estimate_k)
        state_estimate_k_minus = optimal_state_estimate_k
        covariance_estimate_k_minus = covariance_estimate_k
        imu_acc_k_minus = np.array(filter_acc)
        imu_omega_k_minus = np.array(filter_omega)
        time = time + dt
        delta_time_stepping += dt
        if all(isStep[-1]):
            delta_time_stepping += dt
            if delta_time_stepping > 0.04:
                reset_commend = True
                delta_time_stepping = 0
        else:
            reset_commend = False
            delta_time_stepping = 0

    FR = np.zeros((len(x), 3))
    FL = np.zeros((len(x), 3))
    RR = np.zeros((len(x), 3))
    RL = np.zeros((len(x), 3))
    is_walking = np.zeros((len(x), 1))
    num_of_step_leg = np.zeros((len(x), 1))
    for i in range(len(x)):
        FR[i, :] = Pos_in_base[i][0, 0], Pos_in_base[i][0, 1], Pos_in_base[i][0, 2]
        FL[i, :] = Pos_in_base[i][1, 0], Pos_in_base[i][1, 1], Pos_in_base[i][1, 2]
        RR[i, :] = Pos_in_base[i][2, 0], Pos_in_base[i][2, 1], Pos_in_base[i][2, 2]
        RL[i, :] = Pos_in_base[i][3, 0], Pos_in_base[i][3, 1], Pos_in_base[i][3, 2]
        is_walking[i] = ~np.all(isStep[i])
        num_of_step_leg[i] = np.sum(isStep[i])

    # plots
    real_x = real_data[0:-1, 1]
    real_y = real_data[0:-1, 2]
    real_z = real_data[0:-1, 3]
    real_vx = diff(real_x) / dt
    real_vy = diff(real_y) / dt
    real_vz = diff(real_z) / dt
    # real_euler = euler_from_quaternion(Quaternion(real_data[1:-1, 4:8]))

    all_walking_ind = np.where(is_walking)
    start_walking_ind = all_walking_ind[0][0]

    err_x = real_x[0:len(x)] - x
    err_y = real_y[0:len(y)] - y
    err_z = real_z[0:len(z)] - z
    total_err = np.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('total err(% s)' % total_err[-1])
    ax1.plot(t, x, label='x')
    ax1.plot(t, err_x, '-.', label='err x')
    ax1.plot(t, real_x[0:len(x)], '--', label='real_x')
    ax1.text(0, x[-1], 'err(% s)' % err_x[-1], fontsize=12)
    ax1.scatter(t[start_walking_ind], 0, label='start walk')
    ax1.grid()
    ax1.legend()

    ax2.plot(t, y, label='y')
    ax2.plot(t, real_y[0:len(y)], '--', label='real_y')
    ax2.plot(t, err_y, '-.', label='err y')
    ax2.text(0, y[-1], 'err(% s)' % err_y[-1], fontsize=12)
    ax2.scatter(t[start_walking_ind], 0, label='start walk')
    ax2.grid()
    ax2.legend()

    ax3.plot(t, z, label='z')
    ax3.plot(t, real_z[0:len(z)], '--', label='real_z')
    ax3.plot(t, err_z, '-.', label='err z')
    ax3.text(0, z[-1], 'err(% s)' % err_z[-1], fontsize=12)
    ax3.scatter(t[start_walking_ind], 0, label='start walk')
    ax3.grid()
    ax3.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='x')
    plt.plot(t, y, '--', label='y')
    plt.plot(t, z, '-.', label='z')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, vx, label='vx')
    plt.plot(t, vy, '--', label='vy')
    plt.plot(t, vz, '-.', label='vz')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, acc_after_bias_x, label='acc_after_bias_x')
    plt.plot(t, acc_after_bias_y, '--', label='acc_after_bias_y')
    plt.plot(t, acc_after_bias_z, '-.', label='acc_after_bias_z')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, vx, label='vx')
    plt.plot(t, vx_mes, '--', label='vx_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, vy, label='vy')
    plt.plot(t, vy_mes, '--', label='vy_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, vz, label='vz')
    plt.plot(t, vz_mes, '--', label='vz_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    # plt.figure()
    # N = len(acc_z)
    # T = dt
    # Frequency = fftfreq(N, T)[:N // 2]
    # Amplitude_acc_x = fft(acc_x)
    # Amplitude_acc_y = fft(acc_y)
    # Amplitude_acc_z = fft(acc_z)
    # plt.subplot(3, 1, 1)
    # plt.plot(Frequency, 2.0 / N * np.abs(Amplitude_acc_x[0:N // 2]), label='fft acc x')
    # plt.ylabel("Amplitude")
    # plt.xlabel("Frequency [Hz]")
    # plt.legend()
    # plt.grid()
    # plt.subplot(3, 1, 2)
    # plt.plot(Frequency, 2.0 / N * np.abs(Amplitude_acc_y[0:N // 2]), label='fft acc y')
    # plt.ylabel("Amplitude")
    # plt.xlabel("Frequency [Hz]")
    # plt.legend()
    # plt.grid()
    # plt.subplot(3, 1, 3)
    # plt.plot(Frequency, 2.0 / N * np.abs(Amplitude_acc_z[0:N // 2]), label='fft acc z')
    # plt.ylabel("Amplitude")
    # plt.xlabel("Frequency [Hz]")
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, filter_acc_x, label='filter accx')
    plt.plot(t, acc_after_bias_x, '--', label='acc after bias x')
    plt.plot(t, acc_x, '-.', label='accx')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, filter_acc_y, label='filter accy')
    plt.plot(t, acc_after_bias_y, '--', label='acc after bias y')
    plt.plot(t, acc_y, '-.', label='accy')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, filter_acc_z, label='filter accz')
    plt.plot(t, acc_after_bias_z, '--', label='acc after bias z')
    plt.plot(t, acc_z, '-.', label='accz')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='x')
    plt.plot(t, x_imu, '--', label='x_imu')
    plt.plot(t, real_x[0:len(x)], '2', label='real_x')
    plt.plot(t, x_mes, '-.', label='x_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, y, label='y')
    plt.plot(t, y_imu, '--', label='y_imu')
    plt.plot(t, real_y[0:len(y)], '2', label='real_y')
    plt.plot(t, y_mes, '-.', label='y_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, z, label='z')
    plt.plot(t, z_imu, '--', label='z_imu')
    plt.plot(t, real_z[0:len(z)], '2', label='real_z')
    plt.plot(t, z_mes, '-.', label='z_mes')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, vx, label='vx')
    plt.plot(t, vx_imu, '--', label='vx_imu')
    plt.plot(t, vx_mes, '-.', label='vx_mes')
    plt.plot(t, -real_vx[0:len(x)], '2', label='real_vx')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, vy, label='vy')
    plt.plot(t, vy_imu, '--', label='vy_imu(no ekf)')
    plt.plot(t, vy_mes, '-.', label='vy_mes')
    plt.plot(t, real_vy[0:len(y)], '2', label='real_vy')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, vz, label='vz')
    plt.plot(t, vz_imu, '--', label='vz_imu')
    plt.plot(t, vz_mes, '-.', label='vz_mes')
    plt.plot(t, real_vz[0:len(z)], '2', label='real_vz')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    # plt.plot(t, num_of_step_leg, label='number of stepping legs')
    plt.grid()
    plt.legend()
    plt.pause(100000000000)
    # # plt.show
    #
    # plt.figure()
    # ax = p3.Axes3D(fig)
    # # time_vec = np.where(t > 5)
    # # time_vec = time_vec[0]
    # i = 0
    # while i < len(x):
    #     # xsq = [1, 0, 3, 4]
    #     # ysq = [0, 5, 5, 1]
    #     # zsq = [1, 3, 4, 0]
    #     # vertices = [list(zip(x, y, z))]
    #     # poly = Poly3DCollection(vertices, alpha=0.8)
    #     # ax.add_collection3d(poly)
    #
    #     line1 = art3d.Line3D([FR[i, 0] + x[i], x[i]], [FR[i, 1] + y[i], y[i]],
    #                          [FR[i, 2] + z[i], z[i]], color='c')
    #     ax.add_line(line1)
    #     line2 = art3d.Line3D([FL[i, 0] + x[i], x[i]], [FL[i, 1] + y[i], y[i]],
    #                          [FL[i, 2] + z[i], z[i]], color='r')
    #     ax.add_line(line2)
    #     line3 = art3d.Line3D([RR[i, 0] + x[i], x[i]], [RR[i, 1] + y[i], y[i]],
    #                          [RR[i, 2] + z[i], z[i]], color='b')
    #     ax.add_line(line3)
    #     line4 = art3d.Line3D([RL[i, 0] + x[i], x[i]], [RL[i, 1] + y[i], y[i]],
    #                          [RL[i, 2] + z[i], z[i]], color='g')
    #     ax.add_line(line4)
    #     ax.scatter(x[i], y[i], z[i], label='com')
    #     ax.set_xlim(x[0] - 1, x[-1] + 1)
    #     ax.set_ylim(-5, 5)
    #     ax.set_zlim(-z[0], z[-1])
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     ax.text2D(0.05, 0.7, 'pos(% s, %s, %s)' % (x[i], y[i], z[i]),
    #               transform=ax.transAxes)
    #     ax.text2D(0.05, 0.6, 'V:(% s, %s, %s)' % (vx[i], vy[i], vz[i]),
    #               transform=ax.transAxes)
    #     ax.text2D(0.05, 0.5, 'acc:(% s, %s, %s)' % (acc_after_bias_x[i], acc_after_bias_y[i], acc_after_bias_z[i]),
    #               transform=ax.transAxes)
    #     ax.text2D(0.05, 0.4, 'R:(% s)' % ([R[i]]),
    #               transform=ax.transAxes)
    #     ax.text2D(0.05, 0.8,
    #               'step:(FR:% s,FL: %s,RR: %s,RL: %s)' % (isStep[i][0], isStep[i][1], isStep[i][2], isStep[i][3]),
    #               transform=ax.transAxes)
    #     ax.text2D(0.05, 0.9, "TIME:% s" % round(t[i], 4), transform=ax.transAxes)
    #     ax.view_init(elev=25, azim=-60)
    # plt.pause(0.0000001)
    #     ax.cla()
    #     i += 6
