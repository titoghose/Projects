#include <ros/ros.h>
#include <turtlesim/Spawn.h>
#include <ros/console.h>

using namespace ros;

int main(int argc, char **argv){

	init(argc, argv, "spawn_turtle");
	ros::NodeHandle node;
   
    ros::service::waitForService("spawn");
    ros::ServiceClient add_turtle = node.serviceClient<turtlesim::Spawn>("spawn");
  	turtlesim::Spawn::Request req;
  	turtlesim::Spawn::Response res;
  	req.x = 5.5;
  	req.y = 5.5;
  	req.theta = 0;
  	req.name = "master";
    
    if(add_turtle.call(req,res))
    	ROS_INFO("Spawned: ");
    else
    	ROS_INFO("Failed to spawn");
   
  	return 0;
}
