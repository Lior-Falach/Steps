#!/usr/bin/env python3

from time import time
import numpy as np
import rospy
from rospy import Publisher, Rate
from std_msgs.msg import String
from unitree_legged_msgs.msg import A1LowState
from a1 import A1


class A1Interface:

    def __init__(self, move_speed: float = 0.2, rate: int = 10):

        # Init node
        rospy.init_node('robot_state', anonymous=True)

        # Node's attributes
        self._a1: A1 = A1()
        self._rate: Rate = Rate(rate)  # hz
        self._move_speed: float = move_speed
        self._send_cmd: bool = False
        self._cmd: np.array = np.array([2, 0, 0, 0, 0, 0, 0, 0])

        # Low state publisher
        # todo: queue_size = ?
        self._low_state_publisher: Publisher = Publisher('/low_state', A1LowState, queue_size=10)

        # Keyboard subscriber
        rospy.Subscriber('keypress', String, self._update_cmd)

    def _update_cmd(self, data):
        """ The method update the cmd attr according to the keyboard pressed button"""

        # Get the key name from data
        key = data.data.lower()

        # If legal command turn on the `send_cmd` flag
        if key in ['up', 'down', 'right', 'left']:
            self._send_cmd = True

        if key == 'up':

            self._cmd = np.array([2, self._move_speed, 0, 0, 0, 0, 0, 0])

        elif key == 'down':

            self._cmd = np.array([2, -self._move_speed, 0, 0, 0, 0, 0, 0])

        # todo: add right & left

        else:

            self._cmd = np.array([2, 0, 0, 0, 0, 0, 0, 0])

    def run(self):
        """ Run the node """

        while not rospy.is_shutdown():

            # Publish the IMU state
            self._low_state_publisher.publish(*self._a1.low_state())

            # Send robot movement command
            if self._send_cmd:
                print(self._cmd)
                self._a1.high_command(self._cmd)
                self._send_cmd = False

            # Sleep
            self._rate.sleep()


if __name__ == '__main__':

    a1_interface = A1Interface(rate=10)

    try:
        a1_interface.run()
    except rospy.ROSInterruptException:
        pass
