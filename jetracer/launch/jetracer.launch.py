import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    pkg_share = get_package_share_directory('jetracer')
    world_file = os.path.join(pkg_share, 'worlds', 'Trapezoid', 'worlds', 'Trapezoid.world')
    model_path = '/home/developer/ros2_ws/src/car_package/ackermann-vehicle-gzsim-ros2/saye_description/models'

    # Optional features
    enable_green_mask_arg = DeclareLaunchArgument(
        'enable_green_mask', default_value='false',
        description='Enable green mask publisher (make_gt) and viewer')
    
    # Set environment variable
    env_var = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=f"{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}:{model_path}"
    )
    
    # Gazebo simulation
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen',
    )
    
    # Parameter bridge (start after 3 seconds to let Gazebo initialize)
    parameter_bridge = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/rs_front/image@sensor_msgs/msg/Image@gz.msgs.Image',
                    '/rs_front/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
                    '/rs_front/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                    '/rs_front/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
                    '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                    '/model/saye_1/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                    '/chase_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
                ],
                output='screen'
            )
        ]
    )
    # rqt_image_view for front camera (start after 7 seconds)
    rqt_image_front = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                arguments=['/rs_front/image'],
                name='rqt_image_view_front',
                output='screen'
            )
        ]
    )

    green_mask_publisher = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='jetracer',
                executable='make_gt',
                name='green_mask_publisher',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_green_mask'))
            )
        ]
    )

    rqt_image_mask = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                arguments=['/chase_camera/green_mask'],
                name='rqt_image_view_green_mask',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_green_mask'))
            )
        ]
    )

    # start after 15 seconds to let bridge initialize
    management_move = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='jetracer',
                executable='ros_gazebo_ex',
                name='management_move',
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        enable_green_mask_arg,
        env_var,
        gz_sim,
        parameter_bridge,
        rqt_image_front,
        green_mask_publisher,
        rqt_image_mask,
        management_move,
    ])