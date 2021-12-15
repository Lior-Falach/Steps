import numpy as np
from numpy import linalg as LA, ndarray
import math
from pyquaternion import Quaternion
import random
import matplotlib.pyplot as plt

dt = 0.002
I_3 = np.eye(3, dtype=float)
process_noise_v_k_minus_1 = np.random.normal(0, 1, size=(1, 3))*0.001

# State model noise covariance matrix Q_k
Q = np.random.normal(0, 1, size=(6, 6))*0.001

# Measurement matrix H_k
H = np.zeros((3, 15))
H[:, 6:9] = I_3

# Sensor noise
sensor_noise_w_k = np.random.normal(0, 1, size=(1, 3))*0.001

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
Trunk_offset_z = 0.01675  # robot z length(hight)
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


def ekf(z_k_observation_vector, state_estimate_k,
        covariance_estimate_k_minus_1, P_k_v, A_k_minus_1, W_k_minus_1):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.
    OUTPUT
        return state_estimate_k near-optimal state estimate at time k
    """

    ######################### Predict #############################
    # Predict the state estimate at time k based on the state
    # estimate at time k-1 and the control input applied at time k-1.

    print("State Estimate Before EKF=", state_estimate_k)

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    covariance_estimate_k = A_k_minus_1 @ covariance_estimate_k_minus_1.T @ A_k_minus_1.T + W_k_minus_1 @ Q @ W_k_minus_1.T

    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector.T - (H @ state_estimate_k)

    print("Observation=", z_k_observation_vector)

    # Calculate the measurement residual covariance
    S_k = H @ covariance_estimate_k @ H.T + P_k_v

    # Calculate the near-optimal Kalman gain
    # We use pseudocode since some matrices might be
    # non-square or singular.
    K_k = covariance_estimate_k @ H.T @ np.linalg.pinv(S_k)

    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

    # Update the state covariance estimate for time k
    covariance_estimate_k = covariance_estimate_k - (K_k @ H @ covariance_estimate_k)

    # Print the best (near-optimal) estimate of the current state of the robot
    print("State Estimate After EKF=", state_estimate_k)

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


def read_sensor(LowState, state_estimate_k_minus_1, covariance_estimate_k_minus_1):
    z_k = np.array([[0.0, 0.0, 0.0]])
    b_a = np.array([[0.0, 0.0, 0.0]])  # acc noise
    b_w = np.array([[0.0, 0.0, 0.0]])  # omega noise
    b_q = Quaternion()  # q noise
    collect_v_noise = np.array([[0.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, -9.81]])
    r = np.array([[0.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0]])
    q_k = Quaternion()  # unit initial quaternion (q_0)
    rotation_matrix = np.zeros((3, 3))
    euler_rotation = np.array([[0.0, 0.0, 0.0]])  # euler for state

    servo_Angle_mes = np.zeros((4, 3))
    angular_velocity_mes = np.zeros((4, 3))

    state_estimate_k = state_estimate_k_minus_1

    STEP_factor = 0.8
    F_Z_stepping = A1_mass/4*STEP_factor  # my guess for trash hold F=mg/4*factor
    allStep = 0
    Foot_Force_mes = np.array([0.0, 0.0, 0.0, 0.0])
    isStep = np.array([0.0, 0.0, 0.0, 0.0])  # boll vector that show if each leg is stepping
    v_from_leg_k = np.zeros((4, 3))
    v_leg_odometry = np.array([0.0, 0.0, 0.0])

    # read low state massage
    # Measurement
    imu_acc = LowState.imu.accelerometer
    imu_omega = LowState.imu.gyroscope
    q_mes = LowState.imu.quaternion
    imu_quaternion = Quaternion(q_mes[0])

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

    for i in range(4):  # 4 legs
        for j in range(3):  # 3 arms
            servo_Angle_mes[i, :] = ([Hip_angle[i], Thigh_angle[i], Calf_angle[i]])
            angular_velocity_mes[i, :] = (
                [Hip_Angular_velocity[i], Thigh_Angular_velocity[i], Calf_Angular_velocity[i]])

    FootForce_k_minus_1 = Foot_Force_mes
    Foot_Force_mes = LowState.footForce
    isStep = Foot_Force_mes >= F_Z_stepping  # which leg is stepping
    num_of_step_leg = sum(isStep)  # number of leg that stepping

    # Update
    Pos_in_shoulder = forward_kinematics(servo_Angle_mes)
    print("Pos_in_shoulder:", Pos_in_shoulder)
    Pos_in_base = shoulder_2_base(Pos_in_shoulder)
    print("Pos_in_base:", Pos_in_base)

    imu_quaternion = imu_quaternion - b_q
    acc_body = imu_acc - b_a  # In body frame
    omega = imu_omega - b_w

    rotation_angle = dt * LA.norm(omega[0])
    rotation_axis = omega[0] / LA.norm(omega[0])
    # quaternion update
    # Rotate in rotation_angle about rotation_axis, when rotation_angle is in radians
    q_update = Quaternion(axis=rotation_axis, angle=rotation_angle)
    q_k = q_update * q_k
    acc_world = q_k.rotate(acc_body[0])

    rotation_matrix = q_k.rotation_matrix  # quaternion to rotation_matrix of base to world
    euler_rotation = euler_from_quaternion(q_k)
    # acc_world=rotation_matrix@acc_body
    v = v + dt * (acc_world + g)
    r = r + dt * v + math.pow(dt, 2) / 2 * (acc_world + g)

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
    state_estimate_k = state_estimate_k + f_c * dt
    A_k_minus_1 = np.eye(15, dtype=float) + A_c * dt
    W_k_minus_1 = W_c * dt

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

    # covariance for the velocity measurement P_k calculate
    delta_v = v_leg_odometry - v_from_leg_k[isStep]

    D_k = 1 / num_of_step_leg * (delta_v.T@delta_v)
    delta_force = 1 / num_of_step_leg * sum(abs(Foot_Force_mes - FootForce_k_minus_1))
    alpha = 1  # guess
    P_0_v = np.zeros((3, 3))  # my guess
    P_k_v = P_0_v + np.power((0.5 * D_k + I_3 * delta_force / alpha), 2)

    # # Calculate error dynamics matrix
    # quaternion_derivative = rotation_matrix.T @ skew_symmetric_matrix(acc_body)
    # gamma_0 = expm(dt * skew_symmetric_matrix(omega))
    # print(gamma_0)
    # gamma_0 = gamma(dt * skew_symmetric_matrix(omega))
    # print(gamma_0)
    # gamma_1 = gamma_0  # not true you need to fix it
    #
    # F_k = np.array( [[I_3, dt * I_3, -math.pow(dt, 2) / 2.0 * quaternion_derivative, -math.pow(dt, 2) / 2.0 *
    # rotation_matrix.T, 0.0], [0, I_3, -dt * quaternion_derivative, -dt * rotation_matrix.T, 0.0], [0.0, 0.0,
    # gamma_0.T, 0.0, -gamma_1], [0.0, 0.0, 0.0, I_3, 0.0], [0.0, 0.0, 0.0, 0.0, I_3]])
    #
    # print("F_k:", F_k)
    # Q_k = 2
    # W_k = 3
    # print("Time step k=", k)

    #  Run the Extended Kalman Filter
    optimal_state_estimate_k, covariance_estimate_k = ekf(
        z_k,  # Most recent sensor measurement
        state_estimate_k,  # Our most recent estimate of the state
        covariance_estimate_k_minus_1, P_k_v, A_k_minus_1, W_k_minus_1)  # Our most recent state covariance matrix)

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


class Imu:
    def __init__(self):
        self.accelerometer = np.random.rand(1, 3)*9.81
        self.gyroscope = np.random.rand(1, 3)
        self.quaternion = Quaternion.random()

    def __repr__(self):
        return "accelerometer:% s gyroscope:% s quaternion:% s" % (self.accelerometer, self.gyroscope,
                                                                   self.quaternion)


class MotorState:
    def __init__(self):
        self.q = random.uniform(0.0, 1.0)
        self.dq = random.uniform(0.0, 1.0)

    def __repr__(self):
        return "q:% s dq:% s" % (self.q, self.dq)


class LowState:
    def __init__(self):
        self.imu = Imu()
        motorState = []
        for i in range(20):
            motorState.append(MotorState())
        self.motorState = motorState
        self.footForce = np.random.randint(2, size=4)*A1_mass/4
        is_all_zero = np.all((self.footForce == 0))
        if is_all_zero:
            self.footForce = self.footForce + A1_mass/4

    def __repr__(self):
        return "imu:% s motorState:% s footForce:% s" % (self.imu, self.motorState, self.footForce)


if __name__ == '__main__':
    dt = 0.02
    time = 0
    x = []
    y = []
    # FOR FIRST ITERATION
    state_estimate_k_minus_1 = np.zeros((15, 1))
    covariance_estimate_k_minus_1 = np.random.normal(0, 1, size=(15, 15)) * 0.001
    while time < 1:
        lowState = LowState()  # read sensors
        optimal_state_estimate_k, covariance_estimate_k = read_sensor(lowState, state_estimate_k_minus_1, covariance_estimate_k_minus_1)
        x.append(optimal_state_estimate_k[0][0])
        y.append(optimal_state_estimate_k[1][0])
        print()
        print()
        print("time:", time)
        print('optimal_state_estimate_k: ', optimal_state_estimate_k)
        print('covariance_estimate_k: ', covariance_estimate_k)
        state_estimate_k_minus_1 = optimal_state_estimate_k
        covariance_estimate_k_minus_1 = covariance_estimate_k
        time = time + dt
    t = np.arange(0., time-dt, dt)
    plt.plot([x], [y], 'bo')
    plt.show()