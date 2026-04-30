from dataclasses import dataclass
import signal
import time

import rclpy
from capstone_interfaces.msg import SoccerDetections, TrackingStatus
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String


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
        self.declare_parameter('mode_topic', '/soccer/tracking_mode')
        self.declare_parameter('status_topic', '/soccer/tracking_status')
        self.declare_parameter('gimbal_topic', '/manual_gimbal_cmd')
        self.declare_parameter('pan_center', 1500.0)
        self.declare_parameter('tilt_center', 1750.0)
        self.declare_parameter('lock_gimbal', True)
        self.declare_parameter('pan_min', 800.0)
        self.declare_parameter('pan_max', 2200.0)
        self.declare_parameter('tilt_min', 1450.0)
        self.declare_parameter('tilt_max', 1900.0)
        self.declare_parameter('scan_step', 4.0)
        self.declare_parameter('center_tolerance_px', 55.0)
        self.declare_parameter('close_area_ball', 50000.0)
        self.declare_parameter('possession_center_tolerance_px', 170.0)
        self.declare_parameter('possession_row_px', 150.0)
        self.declare_parameter('possession_min_area', 3500.0)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('hold_last_target_timeout_sec', 1.8)
        self.declare_parameter('tracking_deadband_px', 35.0)
        self.declare_parameter('pan_track_step', 3.0)

        self.target_class = 'ball'
        self.mode = 'SEARCH'
        self.servo_x = float(self.get_parameter('pan_center').value)
        self.servo_y = float(self.get_parameter('tilt_center').value)
        self.scan_direction = 1.0
        self.detections = {}

        self.gimbal_pub = self.create_publisher(Point, self.get_parameter('gimbal_topic').value, 10)
        self.status_pub = self.create_publisher(TrackingStatus, self.get_parameter('status_topic').value, 10)
        self.create_subscription(SoccerDetections, self.get_parameter('detection_topic').value, self.detections_callback, 10)
        self.create_subscription(String, self.get_parameter('track_topic').value, self.track_target_callback, 10)
        self.create_subscription(String, self.get_parameter('mode_topic').value, self.mode_callback, 10)
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

    def mode_callback(self, msg):
        requested = msg.data.strip().upper()
        if requested:
            self.mode = requested

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

    def stop_outputs(self):
        pan_center = float(self.get_parameter('pan_center').value)
        tilt_center = float(self.get_parameter('tilt_center').value)
        for _ in range(5):
            try:
                self.publish_gimbal(pan_center, tilt_center)
            except Exception:
                break
            time.sleep(0.05)

    def current_pan_error(self):
        pan_center = float(self.get_parameter('pan_center').value)
        return float(self.servo_x - pan_center)

    def compute_possession_candidate(self, visible, target_class, center_x_error, detection_center_y, area):
        if not visible or target_class != 'ball':
            return False
        center_tolerance = float(self.get_parameter('possession_center_tolerance_px').value)
        possession_row_px = float(self.get_parameter('possession_row_px').value)
        possession_min_area = float(self.get_parameter('possession_min_area').value)
        return (
            abs(center_x_error) <= center_tolerance
            and detection_center_y >= possession_row_px
            and area >= possession_min_area
        )

    def apply_detection_status(self, status, detection, visible, include_y_for_center=False):
        target_center_x = detection.frame_width / 2.0
        target_center_y = detection.frame_height / 2.0
        error_x = float(target_center_x - detection.center_x)
        error_y = float(target_center_y - detection.center_y)
        center_tolerance = float(self.get_parameter('center_tolerance_px').value)

        status.visible = visible
        if include_y_for_center:
            status.centered = abs(error_x) <= center_tolerance and abs(error_y) <= center_tolerance
        else:
            status.centered = abs(error_x) <= center_tolerance
        status.in_range = self.target_class == 'ball' and detection.area >= float(self.get_parameter('close_area_ball').value)
        status.error_x = error_x
        status.error_y = error_y
        status.area = float(detection.area)
        status.confidence = float(detection.confidence)
        status.frame_width = int(detection.frame_width)
        status.frame_height = int(detection.frame_height)
        status.possession_candidate = self.compute_possession_candidate(
            visible,
            self.target_class,
            error_x,
            float(detection.center_y),
            float(detection.area),
        )

    def clear_status(self, status):
        status.visible = False
        status.centered = False
        status.in_range = False
        status.possession_candidate = False

    def control_loop(self):
        detection, stale, age = self.current_detection()
        status = TrackingStatus()
        status.stamp = self.get_clock().now().to_msg()
        status.target_class = self.target_class
        status.stale = stale
        status.pan_error = self.current_pan_error()

        if bool(self.get_parameter('lock_gimbal').value):
            self.servo_x = float(self.get_parameter('pan_center').value)
            self.servo_y = float(self.get_parameter('tilt_center').value)
            self.publish_gimbal(self.servo_x, self.servo_y)
            if detection is not None:
                self.apply_detection_status(status, detection, not stale, include_y_for_center=False)
            else:
                self.clear_status(status)
            status.pan_error = 0.0
            self.status_pub.publish(status)
            return

        if self.mode == 'HOLD':
            if detection is not None:
                self.apply_detection_status(status, detection, not stale, include_y_for_center=False)
            else:
                self.clear_status(status)
            self.publish_gimbal(self.servo_x, self.servo_y)
            self.status_pub.publish(status)
            return

        if self.mode == 'CHASE':
            # During chase the chassis owns steering; the gimbal stays where it is
            # and only reports status back to the FSM.
            if detection is not None:
                self.apply_detection_status(status, detection, not stale, include_y_for_center=False)
            else:
                self.clear_status(status)
            self.servo_y = float(self.get_parameter('tilt_center').value)
            self.publish_gimbal(self.servo_x, self.servo_y)
            self.status_pub.publish(status)
            return

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
            status.pan_error = self.current_pan_error()
            self.clear_status(status)
            self.status_pub.publish(status)
            return

        hold_timeout = float(self.get_parameter('hold_last_target_timeout_sec').value)
        if stale and age is not None and age <= hold_timeout:
            self.clear_status(status)
            status.error_x = float((detection.frame_width / 2.0) - detection.center_x)
            status.error_y = float((detection.frame_height / 2.0) - detection.center_y)
            status.area = float(detection.area)
            status.confidence = float(detection.confidence)
            status.frame_width = int(detection.frame_width)
            status.frame_height = int(detection.frame_height)
            status.pan_error = self.current_pan_error()
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
            status.pan_error = self.current_pan_error()
            self.clear_status(status)
            self.status_pub.publish(status)
            return

        target_center_x = detection.frame_width / 2.0
        target_center_y = detection.frame_height / 2.0
        error_x = target_center_x - detection.center_x
        error_y = target_center_y - detection.center_y

        deadband = float(self.get_parameter('tracking_deadband_px').value)
        pan_step = float(self.get_parameter('pan_track_step').value)
        # During acquisition/alignment the gimbal is allowed to walk the target
        # across the frame, but only in small steps so the chassis still has to turn.
        if error_x > deadband:
            self.servo_x = max(float(self.get_parameter('pan_min').value), self.servo_x + pan_step)
        elif error_x < -deadband:
            self.servo_x = min(float(self.get_parameter('pan_max').value), self.servo_x - pan_step)

        self.servo_y = float(self.get_parameter('tilt_center').value)
        self.publish_gimbal(self.servo_x, self.servo_y)

        self.apply_detection_status(status, detection, True, include_y_for_center=True)
        status.pan_error = self.current_pan_error()
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    stop_requested = False

    def handle_exit_signal(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            node.stop_outputs()
        raise KeyboardInterrupt()

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not stop_requested:
            node.stop_outputs()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


