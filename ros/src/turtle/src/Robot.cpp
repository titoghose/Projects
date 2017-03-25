#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

using namespace ros;

Publisher velocity_publisher;

void move(double distance);

int main(int argc, char *argv[])
{
	init(argc,argv,"Robot");
	NodeHandle n;
	
	velocity_publisher = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",10);

	double distance;
	std::cout<<"Distance: ";
	std::cin>>distance;	
	move(distance);
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
	do{
		velocity_publisher.publish(msg);
		double t1 = Time::now().toSec();
		current_distance = 1.0*(t1-t0);
		spinOnce();
		loop_rate.sleep();
	}while(current_distance < distance || ok());
	msg.linear.x = 0.0;
	velocity_publisher.publish(msg);
}