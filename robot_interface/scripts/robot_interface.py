#!/usr/bin/env python3
from time import time
import rospy
from std_msgs.msg import String
from unitree_legged_msgs.msg import IMU
from robot_interface import RobotInterface  # pytype: disable=import-error


def robot_movement(data, args):
    """The function make the robot move"""
    if data.data == 'walk':

        # Init args
        r = args[0]
        speed = 0.2  # scaled between -1 to 1
        moving_time = 2  # seconds

        # Construct the command
        cmd = [2, speed, 0, 0, 0, 0, 0, 0]  # high command for moving the robot

        # Log
        rospy.loginfo(f'Executing command: {cmd}')

        # Execute the command
        start = time()
        while time() - start < moving_time:
            r.send_high_command(cmd)

        # Log
        rospy.loginfo(f'The command have been done successfully')

    else:

        rospy.loginfo('Unrecognized command')


def robot_state():
    # Init the robot interface
    i = RobotInterface()
    i.send_command([0] * 60)

    # Init publisher & node
    pub = rospy.Publisher('imu', IMU, queue_size=10)
    rospy.Subscriber('robot_movement', String, robot_movement, (i, ))
    rospy.init_node('robot_state', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Start node
    while not rospy.is_shutdown():
        # Get the robot's imu state using the interface
        s = i.receive_observation()

        # Publish the IMU state
        # rospy.loginfo('some log message')
        pub.publish(s.imu.quaternion,
                    s.imu.gyroscope,
                    s.imu.accelerometer,
                    s.imu.temperature)

        # Sleep
        rate.sleep()


if __name__ == '__main__':
    try:
        robot_state()
    except rospy.ROSInterruptException:
        pass