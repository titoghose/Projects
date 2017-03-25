#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from random import randint

pub = rospy.Publisher('Tweet_Hillary', String, queue_size=100)


def callback(data):
    print data.data
    rospy.sleep(2)
    pub.publish('hello')

def hillary():
    rospy.Subscriber('Tweet_Trump', String, callback)
    rospy.init_node('Hillary')
    rospy.spin()


if __name__ == '__main__':
    try:
        hillary()
    except rospy.ROSInterruptException:
        pass
