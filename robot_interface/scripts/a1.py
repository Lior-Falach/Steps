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

    def get_quaternion(self) -> np.array:
        """
        Get the current quaternion of the robot.
        The quaternion is normalized as (w, x, y, z).
        Returns:
            Array with shape (4, )
        """

        return self.r.receive_observation().imu.quaternion

    def get_gyroscope(self) -> np.array:
        """
        Get the current gyroscope of the robot.
        The angular velocity in units [rad/s]
        Returns:
            Array with shape (3, )
        """
        return self.r.receive_observation().imu.gyroscope

    def get_accelerometer(self) -> np.array:
        """
        Get the current accelerometer of the robot.
        The accelerometer in units [m / s^2]
        Returns:
            Array with shape (3, )
        """
        return self.r.receive_observation().imu.accelerometer

    def get_temperature(self) -> int:
        return self.r.receive_observation().imu.temperature


# a1.walk(0.8, 3)
# a1.keyboard_control()