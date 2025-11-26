# from camera.get_odometry import OdometrySubscriber
from jetracer.nodes.sensors.get_odometry import OdometrySubscriber


class PositionProgressor:
    def __init__(self, odometry_subscriber):
        self.odometry_subscriber = odometry_subscriber
        self.target_position = (0.8, 0.0)
        self.last_distance_to_target = None

    def _get_distance_to_target(self):
        self.odometry_subscriber.wait_for_odometry()
        x, y = self.odometry_subscriber.get_actual_position()
        tx, ty = self.target_position
        return ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
    
    def vehicle_in_target_area(self):
        distance = self._get_distance_to_target()
        return distance < 0.2  # Próg odległości do celu

    def get_position_progress(self):
        if self.last_distance_to_target is None:
            self.last_distance_to_target = self._get_distance_to_target()
            return 0.0
        progress = self.last_distance_to_target - self._get_distance_to_target()
        self.last_distance_to_target = self._get_distance_to_target()
        return progress
