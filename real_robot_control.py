"""
Test the C++ robot interface.
"""

from robot_interface import RobotInterface  # pytype: disable=import-error
from time import time, sleep
import threading
from pynput import keyboard


class A1:
    """
    Class for controlling A1 robot.
    High command arg:
        [0] = mode, [1] = forwardSpeed, [2] = sideSpeed, [3] = rotateSpeed,
        [4] = bodyHeight, [5] = roll, [6] = pitch, [7] = yaw
    """
    def __init__(self):

        self.r = RobotInterface()

        self._cmd = [0, 0, 0, 0, 0, 0, 0, 0]
        self._connect = False

    def walk(self, speed, moving_time):
        """
        Make the robot walk in a given speed for a given time
        Args:
            speed: walking speed, scaled between -1 to 1.
            moving_time: time to move in the given speed in seconds

        Returns:

        """
        cmd = [2, speed, 0, 0, 0, 0, 0, 0]  # high command for moving the robot
        start = time()

        while time() - start < moving_time:
            self.r.send_high_command(cmd)

    def _send_cmd(self):

        self._connect = True

        while self._connect:

            print(self._cmd)
            self.r.send_high_command(self._cmd)
            sleep(0.5)

    def _on_press(self, key):

        if key == keyboard.Key.esc:
            self._connect = False
            return False

        elif key.name == 'up':

            self._cmd = [2, 0.5, 0, 0, 0, 0, 0, 0]

        elif key.name == 'down':

            self._cmd = [2, -0.8, 0, 0, 0, 0, 0, 0]

        elif key.name == 'right':

            self._cmd = [2, 0, 0, 0.3, 0, 0, 0, 0]

        elif key.name == 'left':

            self._cmd = [2, 0, 0, -0.3, 0, 0, 0, 0]

        else:

            self._cmd = [0, 0, 0, 0, 0, 0, 0, 0]

    def _on_release(self, key):

        self._cmd = [0, 0, 0, 0, 0, 0, 0, 0]

    def keyboard_control(self):
        """
        Control the robot using you keyboard's arrows
        Returns:

        """
        threading.Thread(target=self._send_cmd).start()

        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()
            print('dead')


a1 = A1()
# a1.walk(0.8, 3)
a1.keyboard_control()