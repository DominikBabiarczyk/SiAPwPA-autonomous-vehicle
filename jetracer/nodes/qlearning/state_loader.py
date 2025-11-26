from jetracer.nodes.qlearning.preprocess_sensor import PreprocessSensor
from jetracer.nodes.sensors.get_odometry import OdometrySubscriber

class StateLoader:
    def __init__(self, preprocess_odometry):
        self.preprocess_sensor = PreprocessSensor()
        self.preprocess_odometry = preprocess_odometry

    def get_state(self):
        sensor_data = self.preprocess_sensor.get_sensor_detect()
        self.preprocess_odometry.wait_for_odometry()
        odom_data = self.preprocess_odometry.last_odom

        x, y = self.preprocess_odometry.get_actual_position()
        yaw = self.preprocess_odometry.get_yaw_from_quaternion(odom_data.pose.pose.orientation)
        return {
            'sensor': sensor_data,
            'position': (x, y),
            'yaw': yaw
        }    
