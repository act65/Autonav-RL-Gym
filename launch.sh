#!/bin/bash

#$1 train or test
#$2 agent type (ddpg, ppo, nav-for turtlebot3 navigation package)
#$3 headless or not
#$4 module_index
#$5 save path
#$6 load path

gnome-terminal -e "./launch/launch_env.sh $1 $2 $3"
sleep 4
gnome-terminal -e "./launch/launch_agent.sh $1 $2 $4 $5 $6"
