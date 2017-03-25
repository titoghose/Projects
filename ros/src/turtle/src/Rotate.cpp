#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include <math.h>

using namespace ros;

Publisher velocity_publisher;

void rotate(double angle);

int main(int argc, char *argv[])
{
	init(argc,argv,"Rotate");
	NodeHandle n;
	
	velocity_publisher = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",10);

	double angle;
	std::cout<<"Angle: ";
	std::cin>>angle;	
	rotate(angle);
	
	spin();
}

void rotate(double angle){
	
	angle = angle * (M_PI/180);

	geometry_msgs::Twist msg;

	msg.linear.x = 0;
	msg.linear.y = 0;
	msg.linear.z = 0;

	msg.angular.x = 0;
	msg.angular.y = 0;
	msg.angular.z = 0.87;

	double t0 = Time::now().toSec();
	double current_angle = 0;
	Rate loop_rate(10);

	while(current_angle < angle){
		velocity_publisher.publish(msg);
		double t1 = Time::now().toSec();
		current_angle = 0.87*(t1-t0);
		spinOnce();
		loop_rate.sleep();
	}

	msg.angular.z = 0.0;
	velocity_publisher.publish(msg);
}