import numpy as np
from numpy import linalg as LA, ndarray
import math
from pyquaternion import Quaternion
import random
import matplotlib.pyplot as plt

from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

f = 200
dt = 1 / f
I_3 = np.eye(3, dtype=float)

# State model noise covariance matrix Q_k
Q = np.random.normal(0, 1, size=(6, 6))

# Measurement matrix H_k
H = np.zeros((3, 15))
H[:, 6:9] = I_3

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


def read_sensor(LowState, state_estimate_k_minus, covariance_estimate_k_minus, reset_commend):
    b_q = Quaternion()  # q noise
    collect_v_noise = np.array([[0.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, 0.0]]) # acc mes don't include gravity
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
    imu_acc = np.array(LowState.imu.accelerometer)
    imu_omega = np.array(LowState.imu.gyroscope)
    # q_mes = np.array(LowState.quaternion)
    # imu_quaternion = Quaternion(q_mes)

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

    # Update state
    Pos_in_shoulder = forward_kinematics(servo_Angle_mes)
    Pos_in_base = shoulder_2_base(Pos_in_shoulder)
    # print("Pos_in_base:", Pos_in_base)
    if reset_commend:
        b_a_minus = np.array([imu_acc])
        b_w_minus = np.array([imu_omega])
        acc_body = imu_acc - b_a_minus  # In body frame
        omega = imu_omega - b_w_minus
        q_k = q_k_minus
    else:
        acc_body = imu_acc - b_a_minus  # In body frame
        omega = imu_omega - b_w_minus
        rotation_angle = dt * LA.norm(omega)
        rotation_axis = omega / LA.norm(omega)
        q_update = Quaternion(axis=rotation_axis[0], angle=rotation_angle)
        s = math.sin(rotation_angle / 2);
        x = rotation_axis[0] * s
        y = rotation_axis[1] * s
        z = rotation_axis[2] * s
        w = math.cos(rotation_angle / 2);
        q_update = Quaternion(w,x,y,z)
        q_k = q_update * q_k_minus

    # quaternion update
    # Rotate in rotation_angle about rotation_axis, when rotation_angle is in radians
    rotation_matrix = q_k.rotation_matrix  # quaternion to rotation_matrix of base to world
    euler_rotation = euler_from_quaternion(q_k)
    v = v_minus + dt * (
            (- skew_symmetric_matrix(np.array([imu_omega])) @ v_minus.T).T + (rotation_matrix.T @ g.T).T + acc_body)
    r = r_minus + dt * v_minus
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
        v_from_leg_k[leg] = -J @ angular_velocity_mes[leg, :] - np.cross(omega, Pos_in_base[leg, :]) + collect_v_noise
        v_leg_odometry = v_leg_odometry + v_from_leg_k[leg, :] * (1 * isStep[leg])
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
    # optimal_state_estimate_k, covariance_estimate_k = ekf(
    #     z_k,  # Most recent sensor measurement
    #     state_estimate_k_,  # Our most recent estimate of the state
    #     covariance_estimate_k_minus,  # Our most recent estimate of the cov
    #     P_k_v, A_k, W_k)  # Our most recent state covariance matrix)

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
    covariance_estimate_k = covariance_estimate_k_minus
    optimal_state_estimate_k = state_estimate_k_
    return optimal_state_estimate_k, covariance_estimate_k, Pos_in_base, isStep, rotation_matrix, acc_body


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
    vx = []
    vy = []
    vz = []
    accx = []
    accy = []
    accz = []
    R = []
    Pos_in_base = []
    isStep = []
    # FOR FIRST ITERATION
    reset_commend = False
    delta_time_stepping = 0
    # FOR FIRST ITERATION
    state_estimate_k_minus = np.zeros((15, 1))
    covariance_estimate_k_minus = np.zeros((15, 15))

    temp_data = np.load("/home/tal/catkin_ws/src/Steps/robot_interface/sim_result.npz", allow_pickle=True)
    data = temp_data['arr_0']

    i = 0
    while time < 5:
        q = data[i]
        dq = data[i + 1]
        ddq = data[i + 2]
        tau = data[i + 3]
        footForce = data[i + 4]
        quaternion = data[i + 5]
        gyroscope = data[i + 6]
        accelerometer = data[i + 7]
        i += 8
        lowState = LowState(q, dq, accelerometer, gyroscope, quaternion, footForce)  # read sensors
        optimal_state_estimate_k, covariance_estimate_k, leg_pos, is_step, rotation_matrix, acc_body = read_sensor(
            lowState,
            state_estimate_k_minus,
            covariance_estimate_k_minus,
            reset_commend)
        t.append(time)
        x.append(optimal_state_estimate_k[0][0])
        y.append(optimal_state_estimate_k[1][0])
        z.append(optimal_state_estimate_k[2][0])
        vx.append(optimal_state_estimate_k[6][0])
        vy.append(optimal_state_estimate_k[7][0])
        vz.append(optimal_state_estimate_k[8][0])
        accx.append(accelerometer[0])
        accy.append(accelerometer[1])
        accz.append(accelerometer[2])
        Pos_in_base.append(leg_pos)
        isStep.append(is_step)
        R.append(rotation_matrix)
        print()
        print()
        print("time:", time)
        print('optimal_state_estimate_k: ', optimal_state_estimate_k)
        # print('covariance_estimate_k: ', covariance_estimate_k)
        state_estimate_k_minus = optimal_state_estimate_k
        covariance_estimate_k_minus = covariance_estimate_k
        time = time + dt
        delta_time_stepping += dt
        if all(isStep[-1]):
            delta_time_stepping += dt
            if delta_time_stepping > 0.2:
                reset_commend = True
                delta_time_stepping = 0
        else:
            reset_commend = False
            delta_time_stepping = 0

        reset_commend = False
    # t = np.arange(0., time - dt, dt)

    FR = np.zeros((len(x), 3))
    FL = np.zeros((len(x), 3))
    RR = np.zeros((len(x), 3))
    RL = np.zeros((len(x), 3))
    is_walking = np.zeros((len(x), 1))
    for i in range(len(x)):  # 4 legs
        FR[i, :] = Pos_in_base[i][0, 0], Pos_in_base[i][0, 1], Pos_in_base[i][0, 2]
        FL[i, :] = Pos_in_base[i][1, 0], Pos_in_base[i][1, 1], Pos_in_base[i][1, 2]
        RR[i, :] = Pos_in_base[i][2, 0], Pos_in_base[i][2, 1], Pos_in_base[i][2, 2]
        RL[i, :] = Pos_in_base[i][3, 0], Pos_in_base[i][3, 1], Pos_in_base[i][3, 2]
        is_walking[i] = ~np.all(isStep[i])
    # Draw a line
    all_walking_ind = np.where(is_walking)
    start_walking_ind = all_walking_ind[0][0]
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='x')
    plt.plot(t, y, label='y')
    plt.plot(t, z, label='z')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, vx, label='vx')
    plt.plot(t, vy, label='vy')
    plt.plot(t, vz, label='vz')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, accx, label='accx')
    plt.plot(t, accy, label='accy')
    plt.plot(t, accz, label='accz')
    plt.scatter(t[start_walking_ind], 0, label='start walk')
    plt.grid()
    plt.legend()
    plt.show

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    frames = []
    # time_vec = np.where(t > 0)
    # time_vec = time_vec[0]
    i = 0
    while i < len(x):
        # xsq = [1, 0, 3, 4]
        # ysq = [0, 5, 5, 1]
        # zsq = [1, 3, 4, 0]
        # vertices = [list(zip(x, y, z))]
        # poly = Poly3DCollection(vertices, alpha=0.8)
        # ax.add_collection3d(poly)

        line1 = art3d.Line3D([FR[i, 0] + x[i], x[i]], [FR[i, 1] + y[i], y[i]],
                             [FR[i, 2] + z[i], z[i]], color='c')
        ax.add_line(line1)
        line2 = art3d.Line3D([FL[i, 0] + x[i], x[i]], [FL[i, 1] + y[i], y[i]],
                             [FL[i, 2] + z[i], z[i]], color='r')
        ax.add_line(line2)
        line3 = art3d.Line3D([RR[i, 0] + x[i], x[i]], [RR[i, 1] + y[i], y[i]],
                             [RR[i, 2] + z[i], z[i]], color='b')
        ax.add_line(line3)
        line4 = art3d.Line3D([RL[i, 0] + x[i], x[i]], [RL[i, 1] + y[i], y[i]],
                             [RL[i, 2] + z[i], z[i]], color='g')
        ax.add_line(line4)
        ax.scatter(x[i], y[i], z[i], label='com')
        # ax.set_xlim(x[i] - 1, x[i] + 1)
        # ax.set_ylim(-10, 10)
        # ax.set_zlim(-x[0], x[-1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.text2D(0.05, 0.7, 'pos(% s, %s, %s)' % (x[i], y[i], z[i]),
                  transform=ax.transAxes)
        ax.text2D(0.05, 0.6, 'V:(% s, %s, %s)' % (vx[i], vy[i], vz[i]),
                  transform=ax.transAxes)
        ax.text2D(0.05, 0.5, 'acc:(% s, %s, %s)' % (accx[i], accy[i], accz[i]),
                  transform=ax.transAxes)
        ax.text2D(0.05, 0.4, 'R:(% s)' % ([R[i]]),
                  transform=ax.transAxes)
        ax.text2D(0.05, 0.8,
                  'step:(FR:% s,FL: %s,RR: %s,RL: %s)' % (isStep[i][0], isStep[i][1], isStep[i][2], isStep[i][3]),
                  transform=ax.transAxes)
        ax.text2D(0.05, 0.9, "TIME:% s" % round(t[i], 4), transform=ax.transAxes)
        ax.view_init(elev=25, azim=-60)
        plt.pause(0.0000001)
        ax.cla()
        i += 3
    # s = 10
    # plt.scatter(FR[:, 0], FR[:, 1], label='FR', s=s)
    # plt.scatter(FL[:, 0], FL[:, 1], label='FL', s=s)
    # plt.scatter(RR[:, 0], RR[:, 1], label='RR', s=s)
    # plt.scatter(RL[:, 0], RL[:, 1], label='RL', s=s)
    # plt.scatter(com[:, 0], com[:, 1], label='com', s=s)
    # plt.scatter(FR[0, 0], FR[0, 1], label='first_FR', marker='x', s=s * 5)
    # plt.scatter(FL[0, 0], FL[0, 1], label='first_FL', marker='x', s=s * 5)
    # plt.scatter(RR[0, 0], RR[0, 1], label='first_RR', marker='x', s=s * 5)
    # plt.scatter(RL[0, 0], RL[0, 1], label='first_RL', marker='x', s=s * 5)
    # plt.scatter(com[0, 0], com[0, 1], label='first_com', marker='>', color='blue', s=s * 5)
    # plt.scatter(FR[-1, 0], FR[-1, 1], label='last_FR', marker='^', s=s * 5)
    # plt.scatter(FL[-1, 0], FL[-1, 1], label='last_FL', marker='^', s=s * 5)
    # plt.scatter(RR[-1, 0], RR[-1, 1], label='last_RR', marker='^', s=s * 5)
    # plt.scatter(RL[-1, 0], RL[-1, 1], label='last_RL', marker='^', s=s * 5)
    # plt.scatter(com[-1, 0], com[-1, 1], label='last_com', marker='<', color='red', s=s * 5)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.grid()
    # plt.show()
