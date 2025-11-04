from camera.Qlerning.position_progressor import PositionProgressor
from camera.check_colision import ColisionChecker


class AwardingPrizes:
    def __init__(self, odometry_subscriber):
        self.position_progressor = PositionProgressor(odometry_subscriber)
        self.collision_checker = ColisionChecker()

    def check_and_award(self):
        target = False
        sum_awards = 0.0
        if self.position_progressor.vehicle_in_target_area():
            sum_awards += 400.0
            target = True  
        progress = self.position_progressor.get_position_progress()
        sum_awards += progress * 30.0 
        self.collision_checker.wait_for_odometry()
        collision = self.collision_checker.is_green_in_rect() 
        if collision:
            sum_awards -= 400.0
        sum_awards -= 0.1 
        return sum_awards, collision, target
