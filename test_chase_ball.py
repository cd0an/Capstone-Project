# test_chase.py
# A simplified test script to isolate and tune ball-tracking PID controllers,
# now upgraded with Gimbal Tracking, Smoothing, and RGB LEDs.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from multiprocessing import Process, Queue
import time

# Import TurboPi hardware message types
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState, RGBState, RGBStates

# Import your existing modules
from Vision.vision import vision_worker
from Utils.helpers import PIDController

class BallChaserNode(Node):
    def __init__(self, data_queue):
        super().__init__('ball_chaser_test')
        self.data_queue = data_queue
        
        # ROS2 Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.servo_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.rgb_pub = self.create_publisher(RGBStates, 'ros_robot_controller/set_rgb', 1)
        
        # 50Hz control loop
        self.timer = self.create_timer(0.02, self.control_loop) 
        
        self.latest_payload = {
            "ball": {"detected": False, "x": 0, "y": 0, "area": 0}
        }

        # --- Hardware Variables ---
        self.servo_x = 1500  # Center position for Pan
        self.servo_y = 1500  # Center position for Tilt
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0

        # --- PID Controllers (Gimbal Tracking) ---
        self.pan_pid = PIDController(kp=0.25, ki=0.05, kd=0.009)
        self.tilt_pid = PIDController(kp=0.25, ki=0.05, kd=0.009)
        
        # Target settings
        self.camera_center_x = 256 
        self.camera_center_y = 256 
        self.target_ball_area = 50000 

        # Center the camera and turn on default lights at startup
        self.publish_servo(self.servo_x, self.servo_y)
        self.publish_rgb(255, 255, 255) # White

    # ---------------------------------------------------------
    # HARDWARE HELPER FUNCTIONS
    # ---------------------------------------------------------
    def smooth_value(self, current_value, previous_value, alpha=0.5):
        """Prevents violent jerking by blending the new speed with the old speed."""
        return alpha * current_value + (1 - alpha) * previous_value

    def publish_servo(self, pan_val, tilt_val):
        """Commands the camera gimbal servos."""
        msg = SetPWMServoState()
        state_x = PWMServoState()
        state_x.id, state_x.position, state_x.offset = [2], [int(pan_val)], [0]
        state_y = PWMServoState()
        state_y.id, state_y.position, state_y.offset = [1], [int(tilt_val)], [0]
        msg.state = [state_x, state_y]
        msg.duration = 0.02
        self.servo_pub.publish(msg)

    def publish_rgb(self, r, g, b):
        """Changes the color of the robot's underglow LEDs."""
        msg = RGBStates()
        state1 = RGBState(index=1, red=int(r), green=int(g), blue=int(b))
        state2 = RGBState(index=2, red=int(r), green=int(g), blue=int(b))
        msg.states = [state1, state2]
        self.rgb_pub.publish(msg)

    # ---------------------------------------------------------
    # MAIN CONTROL LOOP
    # ---------------------------------------------------------
    def control_loop(self):
        # Grab the freshest vision data
        if not self.data_queue.empty():
            self.latest_payload = self.data_queue.get()

        twist = Twist()

        # Check if the ball is visible
        if self.latest_payload["ball"]["detected"]:
            self.publish_rgb(0, 0, 255) # BLUE = Actively Tracking
            
            ball_x = self.latest_payload["ball"]["x"]
            ball_y = self.latest_payload["ball"]["y"]
            ball_area = self.latest_payload["ball"]["area"]

            # 1. Update Camera Gimbal (The "Turret")
            pan_output = self.pan_pid.compute(setpoint=self.camera_center_x, measured_value=ball_x)
            tilt_output = self.tilt_pid.compute(setpoint=self.camera_center_y, measured_value=ball_y)
            
            self.servo_x = max(800, min(2200, self.servo_x + pan_output))
            self.servo_y = max(1200, min(1900, self.servo_y - tilt_output))
            self.publish_servo(self.servo_x, self.servo_y)

            # 2. Calculate Chassis Speed
            # Angular speed comes from the pan (left/right) effort of the camera
            raw_angular_z = pan_output * 0.02
            # Linear speed comes from the difference in area
            raw_linear_x = (self.target_ball_area - ball_area) * 0.00001

            # 3. Apply Limits and Smoothing
            smoothed_angular = self.smooth_value(raw_angular_z, self.last_angular_z)
            smoothed_linear = self.smooth_value(raw_linear_x, self.last_linear_x)

            twist.angular.z = max(min(smoothed_angular, 1.0), -1.0)
            twist.linear.x = max(min(smoothed_linear, 0.5), -0.5)

            # Save history
            self.last_angular_z = twist.angular.z
            self.last_linear_x = twist.linear.x
            
            self.get_logger().info(f"Tracking | Turn: {twist.angular.z:.2f} | Drive: {twist.linear.x:.2f}")

        else:
            self.publish_rgb(255, 0, 0) # RED = Target Lost
            
            # Ball lost: Stop the motors
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
            
            self.pan_pid.reset()
            self.tilt_pid.reset()
            
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
        # Failsafe: Stop motors, center camera, turn off lights on shutdown
        node.cmd_pub.publish(Twist())
        node.publish_servo(1500, 1500)
        node.publish_rgb(0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    payload_queue = Queue(maxsize=1)
    
    # Start the vision processing and ROS control in parallel
    vision_process = Process(target=vision_worker, args=(payload_queue,))
    ros_process = Process(target=ros_worker, args=(payload_queue,))

    try:
        print("Starting TurboPi Chase Test (with Gimbal & Smoothing)...")
        vision_process.start()
        ros_process.start()
        vision_process.join()
        ros_process.join()
    except KeyboardInterrupt:
        print("\nShutting down test...")
        vision_process.terminate()
        ros_process.terminate()
        vision_process.join()
        ros_process.join()
        print("Test complete.")