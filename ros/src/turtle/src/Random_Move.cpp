#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include <math.h>
#include <time.h>
#include <ros/console.h>


using namespace ros;

Publisher velocity_publisher;

void rotate(double angle);
void move(double distance);
void random_move();

int main(int argc, char *argv[])
{
	init(argc,argv,"Random_Move");
	NodeHandle n;
	
	velocity_publisher = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel",10);

	random_move();
}

void random_move(){
	srand(time(NULL));
	while(ok()){
		int distance = rand() % 3 + 1;
		move(distance);
		int angle = rand() % 180 ;
		rotate(angle);
	}
}

void move(double distance){
	
	geometry_msgs::Twist msg;

	msg.linear.x = 3.0;
	msg.linear.y = 0;
	msg.linear.z = 0;

	msg.angular.x = 0;
	msg.angular.y = 0;
	msg.angular.z = 0;

	double t0 = Time::now().toSec();
	double current_distance = 0;
	Rate loop_rate(10);

	while(current_distance < distance){
		ROS_INFO("Hello World");
		velocity_publisher.publish(msg);
		double t1 = Time::now().toSec();
		current_distance = 1.0*(t1-t0);
		spinOnce();
		loop_rate.sleep();
	}

	msg.linear.x = 0.0;
	velocity_publisher.publish(msg);

}

void rotate(double angle){
	
	angle = angle * (M_PI/180);

	geometry_msgs::Twist msg;

	msg.linear.x = 0;
	msg.linear.y = 0;
	msg.linear.z = 0;

	msg.angular.x = 0;
	msg.angular.y = 0;
	msg.angular.z = 1.2;

	double t0 = Time::now().toSec();
	double current_angle = 0;
	Rate loop_rate(10);

	while(current_angle < angle){
		velocity_publisher.publish(msg);
		double t1 = Time::now().toSec();
		current_angle = 1.2*(t1-t0);
		spinOnce();
		loop_rate.sleep();
	}

	msg.angular.z = 0.0;
	velocity_publisher.publish(msg);
}
