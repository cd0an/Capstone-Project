# test_chase_ball.py
# A simplified test script to isolate and tune ball-tracking PID controllers,
# stripped of gimbal and RGB dependencies for fixed-camera chassis tracking.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from multiprocessing import Process, Queue
import time

# Import your existing modules
from Vision.vision import vision_worker
from Utils.helpers import PIDController

class BallChaserNode(Node):
    def __init__(self, data_queue):
        super().__init__('ball_chaser_test')
        self.data_queue = data_queue
        
        # ROS2 Publisher for chassis movement
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        
        # 50Hz control loop
        self.timer = self.create_timer(0.02, self.control_loop) 
        
        self.latest_payload = {
            "ball": {"detected": False, "x": 0, "y": 0, "area": 0},
            "goal": {"detected": False, "x": 0, "y": 0, "area": 0},
            "turbopi": {"detected": False, "x": 0, "y": 0, "area": 0}
        }

        # Chassis Movement PIDs
        # TODO: Tune these constants (Kp, Ki, Kd) for your specific floor surface
        self.angular_pid = PIDController(kp=0.005, ki=0.0, kd=0.001)
        self.linear_pid = PIDController(kp=0.0001, ki=0.0, kd=0.00005)
        
        # Target parameters (Center of a 640x480 frame, and a desired bounding box area to stop at)
        self.center_x = 320 
        self.target_area = 50000 # TODO: Adjust this to change how close the robot gets to the ball

    def control_loop(self):
        # 1. Fetch the freshest vision data
        while not self.data_queue.empty():
            try:
                self.latest_payload = self.data_queue.get_nowait()
            except:
                pass

        ball = self.latest_payload["ball"]
        twist = Twist()

        if ball["detected"]:
            # Calculate errors
            # X error determines turning (yaw), Area error determines forward speed
            error_x = self.center_x - ball["x"]
            error_area = self.target_area - ball["area"]

            # Compute PID outputs
            angular_z = self.angular_pid.compute(error_x)
            
            # Only drive forward if we are roughly centered on the ball
            if abs(error_x) < 100: 
                linear_x = self.linear_pid.compute(error_area)
                # Clamp max speed for safety
                linear_x = max(-0.2, min(0.2, linear_x)) 
            else:
                linear_x = 0.0

            # Assign to Twist message
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            
            self.get_logger().info(f"Tracking Ball - X: {ball['x']}, Area: {ball['area']}")

        else:
            # Ball lost: Reset PIDs and stop the chassis
            self.angular_pid.reset()
            self.linear_pid.reset()
            self.get_logger().info("Ball lost. Stopping.")

        # Send the command to the wheels
        self.cmd_pub.publish(twist)

def ros_worker(data_queue):
    """Initializes ROS2 and spins the test node."""
    rclpy.init()
    node = BallChaserNode(data_queue)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Failsafe: Ensure context is still okay before trying to publish
        if rclpy.ok():
            try:
                node.cmd_pub.publish(Twist())
            except Exception:
                pass
            
        node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    payload_queue = Queue(maxsize=1)
    
    # Start the vision processing and ROS control in parallel
    vision_process = Process(target=vision_worker, args=(payload_queue,))
    ros_process = Process(target=ros_worker, args=(payload_queue,))

    try:
        print("Starting TurboPi Chase Test (Fixed Camera Chassis Tracking)...")
        vision_process.start()
        ros_process.start()
        
        vision_process.join()
        ros_process.join()
    except KeyboardInterrupt:
        print("\nGracefully shutting down processes...")
        
        # Wait up to 2 seconds for the children to run their 'finally' shutdown code
        vision_process.join(timeout=2)
        ros_process.join(timeout=2)
        
        # If they are stuck and still alive after 2 seconds, forcefully terminate them
        if vision_process.is_alive():
            vision_process.terminate()
        if ros_process.is_alive():
            ros_process.terminate()
            
        print("Done.")