import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from typing import Tuple


class ManualVehicleCommander(Node):
    def __init__(self):
        super().__init__("manual_override")
        self.pub = self.create_publisher(Twist, '/manual/cmd_vel', 10)
        self.last_linear:float = 0.0
        self.last_angular:float = 0.0
        self.running:bool = False
        self.num_to_key:dict = {"up":0,"down":1,"left":2,"right":3,"F1":-1,"F4":-2}
        self.key_to_num:dict = {0:"up",1:"down",2:"left",3:"right",-1:"F1",2:"F4"}

    @staticmethod
    def key_to_xz(key:int, throttle:float=0.5, steering:float=0.5) -> Tuple[None|float,None|float]:
        if key == 0:
            return (throttle,None)
        elif key == 1:
            return (-throttle,None)
        elif key == 2:
            return (None, -steering)
        elif key == 3:
            return (None, steering)
        else:
            return (None,None)


    def move(self, key:int) -> None:
        if key == -2:
            self.stop()
            self.destroy_node()
        elif key == -1:
            self.running = not self.running
        
        linear_x, angular_z = ManualVehicleCommander.key_to_xz(key=key)
        if linear_x and angular_z:
            self.last_linear = linear_x
            self.last_angular = angular_z
            self.publish(self.last_linear,self.last_angular)
        elif linear_x:
            self.last_linear = linear_x
            self.publish(self.last_linear,self.last_angular)
        elif angular_z:
            self.last_angular = angular_z
            self.publish(self.last_linear,self.last_angular)
        else:
            self.publish(self.last_linear, self.last_angular)

    def publish(self, linear_x:float, angular_z:float, duration:float = 0.02) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

        self.create_timer(duration, #delay before callback 
                          lambda: self.pub.publish(Twist()), #publish zero
                          oneshot=True) #destroy timer afer running once

    def stop(self) -> None:
        self.publish(0,0,duration=0.2)


def main():
    rclpy.init()
    node = ManualVehicleCommander()
    node.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()