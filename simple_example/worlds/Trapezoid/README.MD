zmienna srodowiskowa: 
```export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/developer/ros2_ws/src/simple_example/worlds/tor/Trapezoid```

```export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/developer/ros2_ws/src/simple_example/worlds/Trapezoid/ackermann-vehicle-gzsim-ros2/saye_description/models/```

uruchomienie:
```gz sim simple_example/worlds/Trapezoid/worlds/Trapezoid.world```

bridge do komunikacji topicow z ros2 do gazebo: ```ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist```

```ros2 run ros_gz_bridge parameter_bridge   /rs_front/image@sensor_msgs/msg/Image@gz.msgs.Image   /rs_front/depth_image@sensor_msgs/msg/Image@gz.msgs.Image   /rs_front/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo   /rs_front/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist```

topic uruchamiajacy jazde: ```ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'```


kamera rqt: ```ros2 run rqt_image_view rqt_image_view```


path:
```echo $GZ_SIM_RESOURCE_PATH
/usr/share/gz/gz-sim8/models:/usr/share/gz/gz-sim-8/models:/home/developer/ros2_ws/src/simple_example/worlds/tor/Trapezoid::/home/developer/ros2_ws/src/tor/Trapezoid/models:/home/developer/ros2_ws/src/simple_example/worlds/tor/Trapezoid:/home/developer/ros2_ws/src/simple_example/worlds/tor/Trapezoid```