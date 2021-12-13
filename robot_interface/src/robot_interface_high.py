#!/usr/bin/env python3
from time import time
import rospy
from std_msgs.msg import String
from unitree_legged_msgs.msg import A1HighState, A1control

from a1 import A1


def robot_movement(data, args):
    """The function make the robot move"""
    if data.data == 'walk':

        # Init args
        r: A1 = args[0]
        rospy.loginfo(r)
        speed: float = 0.2  # scaled between -1 to 1
        moving_time: int = 2  # seconds

        # Construct the command
        cmd = [2, speed, 0, 0, 0, 0, 0, 0]  # high command for moving the robot

        # Log
        rospy.loginfo(f'Executing command: {cmd}')

        # Execute the command
        start = time()
        while time() - start < moving_time:
            r.high_command(cmd)

        # Log
        rospy.loginfo(f'The command have been done successfully')

    else:

        rospy.loginfo('Unrecognized command')

class Robot_control():
    def __init__(self):
        self.a1=A1()
        self.cmd=[0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0]
        self.a1.high_command(self.cmd)
        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('Robot_control')
        self.ros_sub = rospy.Subscriber("/robot_control", A1control, self.Trag_update, queue_size=1)
        rospy.loginfo("Subscriber robot control correctly initialized")
        self.ros_pub_state = rospy.Publisher("/high_state", A1HighState, queue_size=1)
        rospy.loginfo("Publisher to high state correctly initialized")
        self.rcv_time=time()
        self.dur_time=0
    def Trag_update(self,message):
        self.rcv_time=time()
        self.dur_time=message.c[8]
        self.cmd=message.c[0:8]
        self.a1.high_command(self.cmd)

    def run(self):

        # --- Set the control rate
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():

            self.ros_pub_state.publish(*self.a1.high_state())
            rospy.loginfo(self.cmd)
            if (time()-self.rcv_time>self.dur_time):
                self.cmd = [0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0]
                self.a1.high_command(self.cmd)

            # Sleep
            rate.sleep()


def robot_state():

    # Declare A1 instance
    a1 = A1()

    cmd = [0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0]
    a1.high_command(cmd)
    # Init keyboard subscriber
    # rospy.Subscriber('keypress', String, robot_movement, (a1, ))

    # Init publisher
    pub = rospy.Publisher('high_state', A1HighState, queue_size=1)

    # Init node
    rospy.init_node('robot_state', anonymous=True)
    rate = rospy.Rate(100)  # 10hz

    # Start node
    while not rospy.is_shutdown():

        # Publish the IMU state
        rospy.loginfo('some log message')
        pub.publish(*a1.high_state())
        a1.high_command(cmd)

        # Sleep
        rate.sleep()


if __name__ == '__main__':
    try:
        RC=Robot_control()
        RC.run()
    except rospy.ROSInterruptException:
        pass
