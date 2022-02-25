from robot_interface import RobotInterface  # pytype: disable=import-error
import numpy as np
class HCMD:
    def __init__(self):
        self.forwardSpeed = 0.0
        self.sideSpeed = 0.0
        self.rotateSpeed = 0.0
        self.bodyHeight = 0.0

        self.mode = 0 # // 0: idle, default
        # stand
        # 1: forced
        # stand
        # 2: walk
        # continuously
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

class A1:
    """
    Class for controlling A1 robot.
    High command arg:
        [0] = mode, [1] = forwardSpeed, [2] = sideSpeed, [3] = rotateSpeed,
        [4] = bodyHeight, [5] = roll, [6] = pitch, [7] = yaw
    """
    def __init__(self):

        # Init robot interface
        self.r = RobotInterface()

        self.cmd=HCMD()
        #self.r.send_command(self.cmd)

    def high_command(self, cmd):
        """
        The method send high command to the robot.
        The shape of cmd need to be (8, ) where element is according to the
        following description:
                    [0] = mode, [1] = forwardSpeed, [2] = sideSpeed, [3] = rotateSpeed,
                    [4] = bodyHeight, [5] = roll, [6] = pitch, [7] = yaw.
        Args:
            cmd: High command to send to the robot, according the abovementioned description.

        Returns:

        """
        self.r.send_command(cmd)

    def high_state(self):
        """
        The method get the low state of the robot, and make it ready for the `LowState.msg`.

        Returns:
            The variables for the LowState message:
                [q, dq, ddq, tau, footForce, quaternion, gyroscope, accelerometer]
        """

        # Get observation
        o = self.r.receive_observation()
        # print(o.footPosition2Body[0])
        accel=[]
        accel.append(o.imu.accelerometer[0])
        accel.append(o.imu.accelerometer[1])
        accel.append(o.imu.accelerometer[2])

        # accel=o.imu.accelerometer[0],o.imu.accelerometer[1],o.imu.accelerometer[2]
        gyro=[]
        gyro.append(o.imu.gyroscope[0])
        gyro.append(o.imu.gyroscope[1])
        gyro.append(o.imu.gyroscope[2])

        # gyro = o.imu.gyroscope[0], o.imu.gyroscope[1], o.imu.gyroscope[2]
        rf_P=[]
        rf_P.append(o.footPosition2Body[0].x)
        rf_P.append(o.footPosition2Body[0].y)
        rf_P.append(o.footPosition2Body[0].z)

        lf_P = []
        lf_P.append(o.footPosition2Body[1].x)
        lf_P.append(o.footPosition2Body[1].y)
        lf_P.append(o.footPosition2Body[1].z)

        rr_P = []
        rr_P.append(o.footPosition2Body[2].x)
        rr_P.append(o.footPosition2Body[2].y)
        rr_P.append(o.footPosition2Body[2].z)

        lr_P = []
        lr_P.append(o.footPosition2Body[3].x)
        lr_P.append(o.footPosition2Body[3].y)
        lr_P.append(o.footPosition2Body[3].z)

        rf_V = []
        rf_V.append(o.footSpeed2Body[0].x)
        rf_V.append(o.footSpeed2Body[0].y)
        rf_V.append(o.footSpeed2Body[0].z)

        lf_V = []
        lf_V.append(o.footSpeed2Body[1].x)
        lf_V.append(o.footSpeed2Body[1].y)
        lf_V.append(o.footSpeed2Body[1].z)

        rr_V = []
        rr_V.append(o.footSpeed2Body[2].x)
        rr_V.append(o.footSpeed2Body[2].y)
        rr_V.append(o.footSpeed2Body[2].z)

        lr_V = []
        lr_V.append(o.footSpeed2Body[3].x)
        lr_V.append(o.footSpeed2Body[3].y)
        lr_V.append(o.footSpeed2Body[3].z)

        footForce= []
        footForce.append(o.footForce[0])
        footForce.append(o.footForce[1])
        footForce.append(o.footForce[2])
        footForce.append(o.footForce[3])
        # # Get q, dq, ddq vectors
        # q, dq, ddq, tau = [], [], [], []
        #
        # for m in o.motorState[:12]:
        #     q.append(m.q_raw)
        #     dq.append(m.dq_raw)
        #     ddq.append(m.ddq_raw)
        #     tau.append(m.tauEst)

        return accel, gyro, rf_P, lf_P, rr_P, lr_P, rf_P, lf_V, rr_P, lr_V, footForce