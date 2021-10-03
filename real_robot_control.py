"""
Test the C++ robot interface.
"""

from robot_interface import RobotInterface  # pytype: disable=import-error
from time import time, sleep
import threading
from pynput import keyboard


class A1:

    def __init__(self):

        self.r = RobotInterface()

        self._cmd = [0, 0, 0, 0, 0, 0, 0, 0]
        self._connect = False

    def walk(self, speed, moving_time):

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

    def _on_release(self, key):

        self._cmd = [0, 0, 0, 0, 0, 0, 0, 0]

    def keyboard_control(self):

        threading.Thread(target=self._send_cmd).start()

        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()
            print('dead')


a1 = A1()
# a1.walk(0.8, 3)
# a1.keyboard_control()