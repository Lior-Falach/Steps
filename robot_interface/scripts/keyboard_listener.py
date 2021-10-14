#!/usr/bin/env python3
from pynput.keyboard import Listener, Key
from std_msgs.msg import String
import rospy


class KeyboardPublisher:

    def __init__(self):
        # Logging the node start
        rospy.loginfo("Starting node keyboard_publisher\nPress `Esc` and then `ctrl + C to kill the node")

        # Initialise the node
        rospy.init_node("keypress_publisher", anonymous=True)

        # Create a publisher to the keypress topic
        self.keyboard_publisher = rospy.Publisher('/keypress', String, queue_size=1)

    def publish_keypress(self, key_press):
        self.keyboard_publisher.publish(key_press)

    def on_press(self, key):
        print(key)
        self.publish_keypress(key)

    def on_release(self, key):
        print(f'{key} release')
        if key == Key.esc:
            return False

    def keyboard_listener(self):
        # Collect events until released
        with Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


if __name__ == '__main__':

    keyboard_publisher = KeyboardPublisher()
    keyboard_publisher.keyboard_listener()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Stopping keyboard_publisher")
