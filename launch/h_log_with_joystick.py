<?xml version="1.0"?>

<launch>
    <arg name="out_path" default="$(env BEBOP_RECORD_PATH)"/>
    <arg name="id" default=""/>

    <!-- <include file="$(find bebop_driver)/launch/bebop_node.launch"/> -->
    <include file="$(find bebop_tools)/launch/joy_teleop.launch"/>
    <node pkg="deepdrone" name="move_camera_down" type="move_camera_down.py" output="screen" />

    <node pkg="rosbag"
          type="record"
          name="rosbag_record"
          args="-a -o $(arg out_path)/$(arg id)"
          output="screen"
    />

</launch>
