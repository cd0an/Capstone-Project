from dataclasses import dataclass

import rclpy
from capstone_interfaces.msg import SoccerDetections, TrackingStatus
from geometry_msgs.msg import Point, Twist
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


class SoccerFSMNode(Node):
    SEARCH_BALL = 'SEARCH_BALL'
    CENTER_BALL = 'CENTER_BALL'
    APPROACH_BALL = 'APPROACH_BALL'
    SEARCH_GOAL = 'SEARCH_GOAL'
    ALIGN_TO_GOAL = 'ALIGN_TO_GOAL'
    KICK_READY = 'KICK_READY'
    KICK = 'KICK'
    RECOVER = 'RECOVER'

    def __init__(self):
        super().__init__('fsm_node')
        self.declare_parameter('detection_topic', '/soccer/detections')
        self.declare_parameter('status_topic', '/soccer/tracking_status')
        self.declare_parameter('track_topic', '/soccer/track_target')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('rgb_topic', '/manual_rgb_cmd')
        self.declare_parameter('ball_center_tolerance_px', 80.0)
        self.declare_parameter('goal_center_tolerance_px', 50.0)
        self.declare_parameter('ball_area_target', 50000.0)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_strafe_speed', 0.18)
        self.declare_parameter('max_turn_speed', 0.45)
        self.declare_parameter('recover_duration_sec', 1.0)
        self.declare_parameter('kick_ready_hold_sec', 0.5)
        self.declare_parameter('kick_duration_sec', 0.45)
        self.declare_parameter('goal_search_timeout_sec', 4.0)
        self.declare_parameter('ball_lost_timeout_sec', 1.0)
        self.declare_parameter('goal_lost_timeout_sec', 1.0)
        self.declare_parameter('ball_memory_timeout_sec', 1.8)
        self.declare_parameter('lost_ball_forward_speed', 0.08)
        self.declare_parameter('lost_ball_turn_gain', 0.0035)

        self.state = self.SEARCH_BALL
        self.state_enter_time = self.now_seconds()
        self.latest_status = TrackingStatus()
        self.detections = {}
        self.track_target = 'ball'
        self.last_ball_seen_time = self.now_seconds()
        self.last_goal_seen_time = self.now_seconds()
        self.last_ball_error_x = 0.0
        self.last_ball_area = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.rgb_pub = self.create_publisher(Point, self.get_parameter('rgb_topic').value, 10)
        self.track_pub = self.create_publisher(String, self.get_parameter('track_topic').value, 10)
        self.create_subscription(SoccerDetections, self.get_parameter('detection_topic').value, self.detections_callback, 10)
        self.create_subscription(TrackingStatus, self.get_parameter('status_topic').value, self.status_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def detections_callback(self, msg):
        now = self.now_seconds()
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

    def status_callback(self, msg):
        self.latest_status = msg
        now = self.now_seconds()
        if msg.target_class == 'ball' and msg.visible and not msg.stale:
            self.last_ball_seen_time = now
            self.last_ball_error_x = float(msg.error_x)
            self.last_ball_area = float(msg.area)
        if msg.target_class == 'goal' and msg.visible and not msg.stale:
            self.last_goal_seen_time = now

    def get_detection(self, class_name, stale_timeout=0.4):
        detection = self.detections.get(class_name)
        if detection is None:
            return None
        if (self.now_seconds() - detection.received_time) > stale_timeout:
            return None
        return detection

    def transition(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f'Transition: {self.state} -> {new_state}')
            self.state = new_state
            self.state_enter_time = self.now_seconds()

    def publish_target(self, target_class):
        if target_class != self.track_target:
            self.track_target = target_class
        msg = String()
        msg.data = target_class
        self.track_pub.publish(msg)

    def publish_rgb(self, red, green, blue):
        msg = Point()
        msg.x = float(red)
        msg.y = float(green)
        msg.z = float(blue)
        self.rgb_pub.publish(msg)

    def proportional(self, error, gain, limit):
        value = error * gain
        return max(-limit, min(limit, value))

    def control_loop(self):
        twist = Twist()
        ball_detection = self.get_detection('ball')
        goal_detection = self.get_detection('goal')
        now = self.now_seconds()
        state_elapsed = now - self.state_enter_time

        if self.state == self.SEARCH_BALL:
            self.publish_target('ball')
            self.publish_rgb(255, 0, 0)
            twist.angular.z = 0.35
            if ball_detection is not None:
                self.last_ball_seen_time = now
                self.transition(self.CENTER_BALL)

        elif self.state == self.CENTER_BALL:
            self.publish_target('ball')
            self.publish_rgb(0, 0, 255)
            lost_ball = (
                self.latest_status.target_class != 'ball' or
                (now - self.last_ball_seen_time) > float(self.get_parameter('ball_lost_timeout_sec').value)
            )
            if lost_ball:
                self.transition(self.RECOVER)
            else:
                tracking_error_x = self.latest_status.error_x if self.latest_status.visible else self.last_ball_error_x
                twist.angular.z = self.proportional(tracking_error_x, 0.003, float(self.get_parameter('max_turn_speed').value))
                if self.latest_status.centered:
                    self.transition(self.APPROACH_BALL)

        elif self.state == self.APPROACH_BALL:
            self.publish_target('ball')
            self.publish_rgb(0, 120, 255)
            ball_memory_age = now - self.last_ball_seen_time
            lost_ball = (
                self.latest_status.target_class != 'ball' or
                ball_memory_age > float(self.get_parameter('ball_memory_timeout_sec').value)
            )
            if lost_ball:
                self.transition(self.RECOVER)
            else:
                tracking_error_x = self.latest_status.error_x if self.latest_status.visible else self.last_ball_error_x
                tracking_area = self.latest_status.area if self.latest_status.visible else self.last_ball_area
                twist.angular.z = self.proportional(
                    tracking_error_x,
                    float(self.get_parameter('lost_ball_turn_gain').value),
                    float(self.get_parameter('max_turn_speed').value),
                )
                if abs(tracking_error_x) < float(self.get_parameter('ball_center_tolerance_px').value):
                    area_error = float(self.get_parameter('ball_area_target').value) - tracking_area
                    twist.linear.x = self.proportional(area_error, 0.00001, float(self.get_parameter('max_linear_speed').value))
                    if twist.linear.x < 0.0:
                        twist.linear.x = 0.0
                elif not self.latest_status.visible:
                    twist.linear.x = float(self.get_parameter('lost_ball_forward_speed').value)
                if self.latest_status.in_range:
                    self.transition(self.SEARCH_GOAL)

        elif self.state == self.SEARCH_GOAL:
            self.publish_target('goal')
            self.publish_rgb(255, 255, 0)
            if goal_detection is not None and self.latest_status.target_class == 'goal' and self.latest_status.visible:
                self.last_goal_seen_time = now
                self.transition(self.ALIGN_TO_GOAL)
            elif state_elapsed > float(self.get_parameter('goal_search_timeout_sec').value):
                self.transition(self.RECOVER)
            else:
                twist.angular.z = 0.25

        elif self.state == self.ALIGN_TO_GOAL:
            self.publish_target('goal')
            self.publish_rgb(255, 165, 0)
            lost_goal = (
                self.latest_status.target_class != 'goal' or
                (now - self.last_goal_seen_time) > float(self.get_parameter('goal_lost_timeout_sec').value)
            )
            if lost_goal:
                self.transition(self.SEARCH_GOAL)
            else:
                twist.linear.y = self.proportional(self.latest_status.error_x, 0.0015, float(self.get_parameter('max_strafe_speed').value))
                if abs(self.latest_status.error_x) < float(self.get_parameter('goal_center_tolerance_px').value):
                    self.transition(self.KICK_READY)

        elif self.state == self.KICK_READY:
            self.publish_target('goal')
            self.publish_rgb(0, 255, 0)
            if state_elapsed >= float(self.get_parameter('kick_ready_hold_sec').value):
                self.transition(self.KICK)

        elif self.state == self.KICK:
            self.publish_target('goal')
            self.publish_rgb(255, 255, 255)
            if state_elapsed < float(self.get_parameter('kick_duration_sec').value):
                twist.linear.x = 0.15
            else:
                self.transition(self.RECOVER)

        elif self.state == self.RECOVER:
            self.publish_target('ball')
            self.publish_rgb(120, 0, 120)
            if state_elapsed >= float(self.get_parameter('recover_duration_sec').value):
                self.transition(self.SEARCH_BALL)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = SoccerFSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
