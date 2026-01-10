class ObstacleAvoidanceDecider:
  def __init__(self, image_width, image_height):
      self.image_width = image_width
      self.image_height = image_height
  
  def decide(self, obstacle_points: list[tuple[float, float]]):
    
    if not obstacle_points:
        return None
    print(obstacle_points)

    
    max_point = max(obstacle_points, key=lambda p: p[1])
    if max_point[1] > self.image_height * 0.2:
        avg_x = sum(x for x, y in obstacle_points) / len(obstacle_points)
        if avg_x < self.image_width / 2:
            return -1
        else:
            return 1
    return None
  
          