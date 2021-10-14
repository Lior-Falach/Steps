from robot_interface import RobotInterface  # pytype: disable=import-error
import numpy as np


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
        self.r.send_command([0] * 60)

    # def get_quaternion(self) -> np.array:
    #     """
    #     Get the current quaternion of the robot.
    #     The quaternion is normalized as (w, x, y, z).
    #     Returns:
    #         Array with shape (4, )
    #     """
    #
    #     return self.r.receive_observation().imu.quaternion
    #
    # def get_gyroscope(self) -> np.array:
    #     """
    #     Get the current gyroscope of the robot.
    #     The angular velocity in units [rad/s]
    #     Returns:
    #         Array with shape (3, )
    #     """
    #     return self.r.receive_observation().imu.gyroscope
    #
    # def get_accelerometer(self) -> np.array:
    #     """
    #     Get the current accelerometer of the robot.
    #     The accelerometer in units [m / s^2]
    #     Returns:
    #         Array with shape (3, )
    #     """
    #     return self.r.receive_observation().imu.accelerometer
    #
    # def get_temperature(self) -> int:
    #     return self.r.receive_observation().imu.temperature

    def high_command(self, cmd: np.array):
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

        self.r.send_high_command(cmd)

    def low_state(self):
        """
        The method get the low state of the robot, and make it ready for the `LowState.msg`.

        Returns:
            The variables for the LowState message:
                [q, dq, ddq, footForce, quaternion, gyroscope, accelerometer]
        """

        # Get observation
        o = self.r.receive_observation()

        # Get q, dq, ddq vectors
        q, dq, ddq = [], [], []

        for m in o.motorState[:12]:
            q.append(m.q)
            dq.append(m.dq)
            ddq.append(m.ddq)

        return q, dq, ddq, o.footForce, o.imu.quaternion, o.imu.gyroscope, o.imu.accelerometer
