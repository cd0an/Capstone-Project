from dataclasses import dataclass

import rclpy
from capstone_interfaces.msg import SoccerDetections, TrackingStatus
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String

from .pid import PIDController


@dataclass
class DetectionSnapshot:
    class_name: str
    center_x: float
    center_y: float
    area: float
    confidence: float
    frame_width: int
    frame_height: int
    received_time: float


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.declare_parameter('detection_topic', '/soccer/detections')
        self.declare_parameter('track_topic', '/soccer/track_target')
        self.declare_parameter('status_topic', '/soccer/tracking_status')
        self.declare_parameter('gimbal_topic', '/manual_gimbal_cmd')
        self.declare_parameter('pan_center', 1500.0)
        self.declare_parameter('tilt_center', 1650.0)
        self.declare_parameter('pan_min', 800.0)
        self.declare_parameter('pan_max', 2200.0)
        self.declare_parameter('tilt_min', 1450.0)
        self.declare_parameter('tilt_max', 1900.0)
        self.declare_parameter('scan_step', 4.0)
        self.declare_parameter('center_tolerance_px', 55.0)
        self.declare_parameter('close_area_ball', 50000.0)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('hold_last_target_timeout_sec', 1.8)
        self.declare_parameter('pan_output_limit', 10.0)
        self.declare_parameter('tilt_output_limit', 6.0)
        self.declare_parameter('tracking_deadband_px', 35.0)
        self.declare_parameter('tilt_track_enabled', False)
        self.declare_parameter('pan_track_step', 3.0)

        self.target_class = 'ball'
        self.servo_x = float(self.get_parameter('pan_center').value)
        self.servo_y = float(self.get_parameter('tilt_center').value)
        self.scan_direction = 1.0
        self.detections = {}

        self.pan_pid = PIDController(kp=0.08, ki=0.0, kd=0.003)
        self.tilt_pid = PIDController(kp=0.05, ki=0.0, kd=0.002)

        self.gimbal_pub = self.create_publisher(Point, self.get_parameter('gimbal_topic').value, 10)
        self.status_pub = self.create_publisher(TrackingStatus, self.get_parameter('status_topic').value, 10)
        self.create_subscription(SoccerDetections, self.get_parameter('detection_topic').value, self.detections_callback, 10)
        self.create_subscription(String, self.get_parameter('track_topic').value, self.track_target_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

    def detections_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        for detection in msg.detections:
            self.detections[detection.class_name] = DetectionSnapshot(
                class_name=detection.class_name,
                center_x=detection.center_x,
                center_y=detection.center_y,
                area=detection.area,
                confidence=detection.confidence,
                frame_width=detection.frame_width,
                frame_height=detection.frame_height,
                received_time=now,
            )

    def track_target_callback(self, msg):
        requested = msg.data.strip()
        if requested and requested != self.target_class:
            self.target_class = requested
            self.pan_pid.reset()
            self.tilt_pid.reset()

    def current_detection(self):
        detection = self.detections.get(self.target_class)
        if detection is None:
            return None, True, None

        now = self.get_clock().now().nanoseconds / 1e9
        age = now - detection.received_time
        stale = age > float(self.get_parameter('stale_timeout_sec').value)
        return detection, stale, age

    def publish_gimbal(self, pan_value, tilt_value):
        msg = Point()
        msg.x = float(pan_value)
        msg.y = float(tilt_value)
        self.gimbal_pub.publish(msg)

    def control_loop(self):
        detection, stale, age = self.current_detection()
        status = TrackingStatus()
        status.stamp = self.get_clock().now().to_msg()
        status.target_class = self.target_class
        status.stale = stale

        if detection is None:
            pan_min = float(self.get_parameter('pan_min').value)
            pan_max = float(self.get_parameter('pan_max').value)
            tilt_center = float(self.get_parameter('tilt_center').value)
            scan_step = float(self.get_parameter('scan_step').value)

            self.servo_x += self.scan_direction * scan_step
            if self.servo_x >= pan_max:
                self.servo_x = pan_max
                self.scan_direction = -1.0
            elif self.servo_x <= pan_min:
                self.servo_x = pan_min
                self.scan_direction = 1.0

            self.servo_y = tilt_center
            self.publish_gimbal(self.servo_x, self.servo_y)
            status.visible = False
            status.centered = False
            status.in_range = False
            self.status_pub.publish(status)
            return

        hold_timeout = float(self.get_parameter('hold_last_target_timeout_sec').value)
        if stale and age is not None and age <= hold_timeout:
            status.visible = False
            status.centered = False
            status.in_range = False
            status.error_x = float((detection.frame_width / 2.0) - detection.center_x)
            status.error_y = float((detection.frame_height / 2.0) - detection.center_y)
            status.area = float(detection.area)
            status.confidence = float(detection.confidence)
            status.frame_width = int(detection.frame_width)
            status.frame_height = int(detection.frame_height)
            self.publish_gimbal(self.servo_x, self.servo_y)
            self.status_pub.publish(status)
            return

        if stale:
            pan_min = float(self.get_parameter('pan_min').value)
            pan_max = float(self.get_parameter('pan_max').value)
            tilt_center = float(self.get_parameter('tilt_center').value)
            scan_step = float(self.get_parameter('scan_step').value)

            self.servo_x += self.scan_direction * scan_step
            if self.servo_x >= pan_max:
                self.servo_x = pan_max
                self.scan_direction = -1.0
            elif self.servo_x <= pan_min:
                self.servo_x = pan_min
                self.scan_direction = 1.0

            self.servo_y = tilt_center
            self.publish_gimbal(self.servo_x, self.servo_y)
            status.visible = False
            status.centered = False
            status.in_range = False
            self.status_pub.publish(status)
            return

        target_center_x = detection.frame_width / 2.0
        target_center_y = detection.frame_height / 2.0
        error_x = target_center_x - detection.center_x
        error_y = target_center_y - detection.center_y

        deadband = float(self.get_parameter('tracking_deadband_px').value)
        pan_step = float(self.get_parameter('pan_track_step').value)
        if error_x > deadband:
            self.servo_x = max(float(self.get_parameter('pan_min').value), self.servo_x + pan_step)
        elif error_x < -deadband:
            self.servo_x = min(float(self.get_parameter('pan_max').value), self.servo_x - pan_step)

        self.servo_y = float(self.get_parameter('tilt_center').value)
        self.publish_gimbal(self.servo_x, self.servo_y)

        center_tolerance = float(self.get_parameter('center_tolerance_px').value)
        ball_close_area = float(self.get_parameter('close_area_ball').value)

        status.visible = True
        status.centered = abs(error_x) <= center_tolerance and abs(error_y) <= center_tolerance
        status.in_range = self.target_class == 'ball' and detection.area >= ball_close_area
        status.error_x = float(error_x)
        status.error_y = float(error_y)
        status.area = float(detection.area)
        status.confidence = float(detection.confidence)
        status.frame_width = int(detection.frame_width)
        status.frame_height = int(detection.frame_height)
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
