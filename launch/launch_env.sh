#!/bin/bash
# to be run from the root dir of this repo
# (should just run the bash script in root instead of running this directly)

cd ../..
source ./devel/setup.bash

echo $3

if [ $3 == "headless" ]; then
	env_name="ing_env_headless.launch"
else
	env_name="ing_env.launch"
fi

echo $1$env_name

if [ $2 == "nav" ]; then
	roslaunch project $1$env_name drive_type:=twist
else
	roslaunch project $1$env_name
fi

# if [ $2 == "nav" ]; then
# 	roslaunch project $1ing_env.launch drive_type:=twist
# else
# 	roslaunch project $1ing_env_headless.launch
# fi

while true
do
	:
done
