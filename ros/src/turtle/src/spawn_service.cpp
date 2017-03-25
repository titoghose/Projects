#include "ros/ros.h"
#include "turtlesim/Spawn.h"
   
   bool spawn(turtlesim::Spawn::Request  &req, turtlesim::Spawn::Response &res)
   {
    	res.name = req.name;
    	return true;
   }

   int main(int argc, char **argv)
   {
     ros::init(argc, argv, "spawn_service");
     ros::NodeHandle n;
   
     ros::ServiceServer service = n.advertiseService("spawn", spawn);
     ros::spin();
   
     return 0;
   }
