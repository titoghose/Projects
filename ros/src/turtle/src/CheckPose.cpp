#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "turtlesim/Pose.h"

using namespace ros;

Publisher velocity_publisher;
Subscriber pose_sub;
turtlesim::Pose curr_pose;

void move(double distance);
void turtleCallBack(const turtlesim::Pose::ConstPtr& msg);


int main(int argc, char *argv[])
{
	init(argc,argv,"Move");
	NodeHandle n;
	
	velocity_publisher = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",10);
	pose_sub = n.subscribe("/turtle1/pose",100,turtleCallBack);
	
	double distance;
	std::cout<<"Distance: ";
	std::cin>>distance;	
	move(distance);
	
	spin();
	
}

void turtleCallBack(const turtlesim::Pose::ConstPtr& msg){
	curr_pose.x = msg->x;
	curr_pose.y = msg->y;
	curr_pose.theta = msg->theta;
	//ROS_INFO_ONCE("%f %f",msg->x,msg->y);
	return;
}

void move(double distance){
	
	geometry_msgs::Twist msg;

	msg.linear.x = 1.0;
	msg.linear.y = 0;
	msg.linear.z = 0;

	msg.angular.x = 0;
	msg.angular.y = 0;
	msg.angular.z = 0;

	double t0 = Time::now().toSec();
	double current_distance = 0;
	Rate loop_rate(10);
	
	
	while(current_distance < distance){
		velocity_publisher.publish(msg);
		double t1 = Time::now().toSec();
		current_distance = 1.0*(t1-t0);
		spinOnce();
		loop_rate.sleep();
	}
	ROS_INFO("%f %f %f",curr_pose.x,curr_pose.y,curr_pose.theta);
	msg.linear.x = 0.0;
	velocity_publisher.publish(msg);
}
