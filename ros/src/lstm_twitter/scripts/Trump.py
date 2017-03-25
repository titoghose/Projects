#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from random import randint

pub = rospy.Publisher('Tweet_Trump', String, queue_size=100)


def callback(data):
    print data.data
    rospy.sleep(2)
    pub.publish("Hi")



def trump():
    rospy.Subscriber('Tweet_Hillary', String, callback)
    rospy.init_node('Trump')
    pub.publish("Hi")
    rospy.spin()


if __name__ == '__main__':
    try:
        trump()
    except rospy.ROSInterruptException:
        pass
