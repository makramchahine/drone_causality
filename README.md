# DeepDrone

## Installation
Setup catkin workspace:

```
mkdir -p ~/drone_ws/src && cd ~/drone_ws
catkin init
```

Download Bebop drivers and DeepDrone repository: 

```
git clone https://github.com/AutonomyLab/bebop_autonomy.git src/bebop_autonomy
git clone git@github.com:mit-drl/deepdrone.git src/deepdrone
rosdep update
rosdep install --from-paths src -i
```

Build the workspace: 
```
catkin build
```

## Setup the workspace
```
source ~/drone_ws/devel/setup
```

## Other useful commands: 
1. Start the Bebop ROS node: `roslaunch bebop_driver bebop_node.launch`
2. Launch the joystick controller: `roslaunch bebop_tools joy_teleop.launch`
3. Point camera down: `rosrun deepdrone talker.py`
4. Record data into rosbag: `rosbag record -a -o recording`
