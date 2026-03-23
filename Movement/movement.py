# movement.py
# Implements all basic and slanted movements (forward, backward, left, right, diagonal)
# and turns for the robot using ROS Twist messages 

import rclpy
import signal
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class LMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = 0.0  
            self.twist.linear.y = 1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class RMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = 0.0  
            self.twist.linear.y = -1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class FMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = 1.0  
            self.twist.linear.y = 0.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class BMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = -1.0  
            self.twist.linear.y = 0.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class LFSlantMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = 1.0  
            self.twist.linear.y = 1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class RFSlantMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = 1.0  
            self.twist.linear.y = -1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class LBSlantMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = -1.0  
            self.twist.linear.y = 1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class RBSlantMovement(Node):
    def __init__(self):
        super().__init__('car_slant')
        signal.signal(signal.SIGINT, self.stop)
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1) # chassis control
        time.sleep(1)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0 # current state
        self.twist = Twist() # movement command

    def timer_callback(self):
        if self.state == 0:
            self.twist.linear.x = -1.0  
            self.twist.linear.y = -1.0

        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: linear.x = {self.twist.linear.x}, lineary.y = {self.twist.linear.y}")

    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class LTurnMovement(Node):
    def __init__(self):
        super().__init__('car_turn') 
        signal.signal(signal.SIGINT, self.stop)  
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1)  # 底盘控制
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0  # 当前状态
        self.twist = Twist()  # 运动命令

    def timer_callback(self):
        if self.state == 0:
            self.twist.angular.z = 10.0 # Left
            
        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: angular.z = {self.twist.angular.z}")
        
    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

class RTurnMovement(Node):
    def __init__(self):
        super().__init__('car_turn') 
        signal.signal(signal.SIGINT, self.stop)  
        self.mecanum_pub = self.create_publisher(Twist, 'cmd_vel', 1)  # 底盘控制
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.state = 0  # 当前状态
        self.twist = Twist()  # 运动命令

    def timer_callback(self):
        if self.state == 0:
            self.twist.angular.z = -10.0 # Right
            
        self.mecanum_pub.publish(self.twist)
        self.get_logger().info(f"Published twist: angular.z = {self.twist.angular.z}")
        
    def stop(self, signum, frame):
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

def main():
    rclpy.init()
    # node =
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.mecanum_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()