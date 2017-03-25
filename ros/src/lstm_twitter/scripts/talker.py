#!/usr/bin/env python
import rospy
from std_msgs.msg import String


def talker():
    # Creating a publisher that will publish on Chatter topic
    pub = rospy.Publisher('Chatter', String, queue_size=10)
    # Creating a node called talker
    rospy.init_node('talker', anonymous=True)
    # setting the frequency of publishing to 1 hz i.e 1 msg per second
    rate = rospy.Rate(1)  # 1 hz
    # till Ctrl C is pressed
    while not rospy.is_shutdown():
        # the message to publish
        str = "Time is %s" % rospy.get_time()
        # prints the message and adds it to talker node's log file
        rospy.loginfo(str)
        # publishing the message to the topic
        pub.publish(str)
        # sleeping for 1 hz
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
