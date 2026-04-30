from dataclasses import dataclass
import signal
import time

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
    ALIGN_TO_BALL = 'ALIGN_TO_BALL'
    APPROACH_BALL = 'APPROACH_BALL'
    BALL_POSSESSION = 'BALL_POSSESSION'
    SEARCH_GOAL = 'SEARCH_GOAL'
    ALIGN_TO_GOAL = 'ALIGN_TO_GOAL'
    DRIVE_TO_GOAL = 'DRIVE_TO_GOAL'
    KICK = 'KICK'
    RECOVER = 'RECOVER'

    def __init__(self):
        super().__init__('fsm_node')
        self.declare_parameter('detection_topic', '/soccer/detections')
        self.declare_parameter('status_topic', '/soccer/tracking_status')
        self.declare_parameter('track_topic', '/soccer/track_target')
        self.declare_parameter('mode_topic', '/soccer/tracking_mode')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('rgb_topic', '/manual_rgb_cmd')
        self.declare_parameter('forward_sign', 1.0)
        self.declare_parameter('turn_sign', 1.0)
        self.declare_parameter('linear_speed_scale', 0.55)
        self.declare_parameter('angular_speed_scale', 0.45)
        self.declare_parameter('max_linear_step', 0.05)
        self.declare_parameter('max_angular_step', 0.12)
        self.declare_parameter('min_effective_linear_speed', 0.10)
        self.declare_parameter('min_effective_turn_speed', 0.10)
        self.declare_parameter('linear_breakaway_speed', 0.26)
        self.declare_parameter('linear_hold_speed', 0.18)
        self.declare_parameter('angular_breakaway_speed', 2.60)
        self.declare_parameter('angular_hold_speed', 1.20)
        self.declare_parameter('approach_turn_breakaway_speed', 1.50)
        self.declare_parameter('approach_turn_hold_speed', 0.45)
        self.declare_parameter('approach_turn_stuck_err_delta_px', 12.0)
        self.declare_parameter('approach_turn_stuck_hold_sec', 0.8)
        self.declare_parameter('approach_turn_stuck_breakaway_speed', 2.20)
        self.declare_parameter('approach_turn_stuck_hold_speed', 0.75)
        self.declare_parameter('chase_angular_hold_speed', 0.18)
        self.declare_parameter('motion_breakaway_duration_sec', 0.10)
        self.declare_parameter('ball_area_target', 50000.0)
        self.declare_parameter('startup_hold_sec', 4.0)
        self.declare_parameter('search_spin_on_sec', 0.12)
        self.declare_parameter('search_spin_off_sec', 1.40)
        self.declare_parameter('max_turn_speed', 3.0)
        self.declare_parameter('recover_duration_sec', 0.8)
        self.declare_parameter('ball_possession_hold_sec', 0.6)
        self.declare_parameter('ball_possession_settle_sec', 0.70)
        self.declare_parameter('ball_possession_settle_speed', 0.14)
        self.declare_parameter('ball_possession_hold_speed', 0.05)
        self.declare_parameter('ball_possession_release_ignore_sec', 1.40)
        self.declare_parameter('ball_possession_release_hold_sec', 0.70)
        self.declare_parameter('ball_possession_release_err_y_min', 10.0)
        self.declare_parameter('ball_possession_release_area_max', 18000.0)
        self.declare_parameter('kick_duration_sec', 0.45)
        self.declare_parameter('goal_search_timeout_sec', 5.0)
        self.declare_parameter('ball_lost_timeout_sec', 1.2)
        self.declare_parameter('goal_lost_timeout_sec', 1.0)
        self.declare_parameter('lost_ball_forward_speed', 0.0)
        self.declare_parameter('lost_ball_turn_gain', 0.0)
        self.declare_parameter('ball_align_turn_gain', 0.015)
        self.declare_parameter('ball_chase_turn_gain', 0.0022)
        self.declare_parameter('goal_align_turn_gain', 0.015)
        self.declare_parameter('goal_drive_speed', 0.28)
        self.declare_parameter('goal_drive_duration_sec', 1.2)
        self.declare_parameter('ball_align_pan_tolerance', 50.0)
        self.declare_parameter('ball_align_timeout_sec', 1.2)
        self.declare_parameter('goal_align_pan_tolerance', 50.0)
        self.declare_parameter('min_align_turn_speed', 0.3)
        self.declare_parameter('min_chase_turn_speed', 0.12)
        self.declare_parameter('ball_chase_center_threshold_px', 90.0)
        self.declare_parameter('ball_close_center_threshold_px', 100.0)
        self.declare_parameter('ball_close_steer_band_px', 150.0)
        self.declare_parameter('ball_close_center_area', 4500.0)
        self.declare_parameter('ball_chase_crawl_threshold_px', 160.0)
        self.declare_parameter('ball_chase_crawl_speed', 0.06)
        self.declare_parameter('ball_close_steer_speed', 0.08)
        self.declare_parameter('ball_chase_max_turn_speed', 0.14)
        self.declare_parameter('ball_close_max_turn_speed', 0.22)
        self.declare_parameter('ball_chase_max_speed', 0.12)
        self.declare_parameter('possession_candidate_hold_sec', 0.05)
        self.declare_parameter('blind_zone_capture_timeout_sec', 0.50)
        self.declare_parameter('possession_turn_tolerance_px', 140.0)
        self.declare_parameter('possession_max_turn_cmd', 0.14)

        self.startup_time = self.now_seconds()
        self.state = self.SEARCH_BALL
        self.state_enter_time = self.startup_time
        self.latest_status = TrackingStatus()
        self.detections = {}
        self.track_target = 'ball'
        self.last_ball_seen_time = self.now_seconds()
        self.last_goal_seen_time = self.now_seconds()
        self.last_ball_error_x = 0.0
        self.last_ball_area = 0.0
        self.last_ball_pan_error = 0.0
        self.last_goal_pan_error = 0.0
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        self.linear_active_since = None
        self.angular_active_since = None
        self.last_possession_candidate_time = 0.0
        self.candidate_stable_since = None
        self.ball_blind_zone_since = None
        self.last_approach_was_straight = False
        self.ball_possession_release_since = None
        self.approach_turn_reference_error = None
        self.approach_turn_stuck_since = None
        self.approach_turn_stuck_active = False

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.rgb_pub = self.create_publisher(Point, self.get_parameter('rgb_topic').value, 10)
        self.track_pub = self.create_publisher(String, self.get_parameter('track_topic').value, 10)
        self.mode_pub = self.create_publisher(String, self.get_parameter('mode_topic').value, 10)
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
            self.last_ball_pan_error = float(msg.pan_error)
        if msg.target_class == 'goal' and msg.visible and not msg.stale:
            self.last_goal_seen_time = now
            self.last_goal_pan_error = float(msg.pan_error)

    def get_detection(self, class_name, stale_timeout=0.4):
        detection = self.detections.get(class_name)
        if detection is None:
            return None
        if (self.now_seconds() - detection.received_time) > stale_timeout:
            return None
        return detection

    def reset_possession_tracking(self):
        self.last_possession_candidate_time = 0.0
        self.candidate_stable_since = None
        self.ball_blind_zone_since = None
        self.last_approach_was_straight = False
        self.ball_possession_release_since = None
        self.approach_turn_reference_error = None
        self.approach_turn_stuck_since = None
        self.approach_turn_stuck_active = False
        self.approach_turn_reference_error = None
        self.approach_turn_stuck_since = None
        self.approach_turn_stuck_active = False

    def transition(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f'Transition: {self.state} -> {new_state}')
            self.state = new_state
            self.state_enter_time = self.now_seconds()
            self.angular_active_since = None
            self.linear_active_since = None
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
            self.reset_possession_tracking()

    def publish_target(self, target_class):
        if target_class != self.track_target:
            self.track_target = target_class
        msg = String()
        msg.data = target_class
        self.track_pub.publish(msg)

    def publish_mode(self, mode):
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)

    def publish_rgb(self, red, green, blue):
        msg = Point()
        msg.x = float(red)
        msg.y = float(green)
        msg.z = float(blue)
        self.rgb_pub.publish(msg)

    def stop_outputs(self):
        zero_twist = Twist()
        leds_off = Point()
        leds_off.x = 0.0
        leds_off.y = 0.0
        leds_off.z = 0.0
        for _ in range(5):
            try:
                self.cmd_pub.publish(zero_twist)
                self.rgb_pub.publish(leds_off)
            except Exception:
                break
            time.sleep(0.05)

    def proportional(self, error, gain, limit):
        value = error * gain
        return max(-limit, min(limit, value))

    def limit_rate(self, target, previous, step_limit):
        delta = target - previous
        if delta > step_limit:
            return previous + step_limit
        if delta < -step_limit:
            return previous - step_limit
        return target

    def enforce_axis_motion_profile(self, target, now, breakaway_speed, hold_speed, active_since_attr):
        if abs(target) < 1e-6:
            setattr(self, active_since_attr, None)
            return 0.0

        active_since = getattr(self, active_since_attr)
        if active_since is None:
            active_since = now
            setattr(self, active_since_attr, active_since)

        elapsed = now - active_since
        floor = breakaway_speed if elapsed < float(self.get_parameter('motion_breakaway_duration_sec').value) else hold_speed
        magnitude = max(abs(target), floor)
        return magnitude if target > 0.0 else -magnitude

    def biased_turn(self, error, gain, limit, min_turn):
        turn = self.proportional(error, gain, limit)
        if abs(error) < 1.0:
            return 0.0
        if abs(turn) < min_turn:
            return min_turn if turn >= 0.0 else -min_turn
        return turn

    def control_loop(self):
        twist = Twist()
        ball_detection = self.get_detection('ball')
        _goal_detection = self.get_detection('goal')
        now = self.now_seconds()
        state_elapsed = now - self.state_enter_time
        forward_sign = float(self.get_parameter('forward_sign').value)
        turn_sign = float(self.get_parameter('turn_sign').value)

        if (now - self.startup_time) < float(self.get_parameter('startup_hold_sec').value):
            self.publish_target('ball')
            self.publish_mode('HOLD')
            self.publish_rgb(0, 0, 0)

        elif self.state == self.SEARCH_BALL:
            self.publish_target('ball')
            self.publish_mode('SEARCH')
            self.publish_rgb(255, 0, 0)
            if ball_detection is not None:
                self.last_ball_seen_time = now
                self.transition(self.APPROACH_BALL)
            else:
                search_on = float(self.get_parameter('search_spin_on_sec').value)
                search_off = float(self.get_parameter('search_spin_off_sec').value)
                search_cycle = search_on + search_off
                if search_cycle > 0.0 and (state_elapsed % search_cycle) < search_on:
                    # Keep searching toward the same side the ball was last seen
                    # on instead of always spinning the same way.
                    search_direction = 1.0
                    if self.last_ball_error_x > 1.0:
                        search_direction = 1.0
                    elif self.last_ball_error_x < -1.0:
                        search_direction = -1.0
                    twist.angular.z = turn_sign * search_direction

        elif self.state == self.ALIGN_TO_BALL:
            self.publish_target('ball')
            self.publish_mode('TRACK')
            self.publish_rgb(0, 0, 255)
            lost_ball = (
                self.latest_status.target_class != 'ball' or
                (now - self.last_ball_seen_time) > float(self.get_parameter('ball_lost_timeout_sec').value)
            )
            if lost_ball:
                self.transition(self.RECOVER)
            else:
                camera_angle_error = self.latest_status.pan_error if self.latest_status.visible else self.last_ball_pan_error
                twist.angular.z = self.biased_turn(
                    camera_angle_error,
                    float(self.get_parameter('ball_align_turn_gain').value),
                    float(self.get_parameter('max_turn_speed').value),
                    float(self.get_parameter('min_align_turn_speed').value),
                )
                twist.angular.z *= turn_sign
                if (
                    abs(camera_angle_error) < float(self.get_parameter('ball_align_pan_tolerance').value) or
                    (state_elapsed >= float(self.get_parameter('ball_align_timeout_sec').value) and self.latest_status.visible)
                ):
                    self.transition(self.APPROACH_BALL)

        elif self.state == self.APPROACH_BALL:
            self.publish_target('ball')
            self.publish_mode('HOLD')
            self.publish_rgb(0, 120, 255)
            possession_turn_tolerance = float(self.get_parameter('possession_turn_tolerance_px').value)
            possession_max_turn_cmd = float(self.get_parameter('possession_max_turn_cmd').value)
            visible_ball = (
                self.latest_status.target_class == 'ball' and
                self.latest_status.visible and
                not self.latest_status.stale
            )
            debug_message = None

            if visible_ball:
                error_x = self.latest_status.error_x
                tracking_area = self.latest_status.area
                error_y = self.latest_status.error_y
                close_area_mode = tracking_area >= float(self.get_parameter('ball_close_center_area').value)
                center_threshold = float(self.get_parameter('ball_close_center_threshold_px').value) if close_area_mode else float(self.get_parameter('ball_chase_center_threshold_px').value)
                centered_enough = abs(error_x) < center_threshold
                capture_aligned = abs(error_x) < possession_turn_tolerance
                self.ball_blind_zone_since = None
                self.ball_possession_release_since = None

                close_steer_band = float(self.get_parameter('ball_close_steer_band_px').value)
                general_steer_band = float(self.get_parameter('ball_chase_crawl_threshold_px').value)
                close_creep_mode = close_area_mode and not centered_enough and abs(error_x) <= close_steer_band
                general_creep_mode = (not close_area_mode) and (not centered_enough) and abs(error_x) <= general_steer_band
                creep_mode = close_creep_mode or general_creep_mode

                if centered_enough:
                    twist.angular.z = 0.0
                    area_error = max(0.0, float(self.get_parameter('ball_area_target').value) - tracking_area)
                    base_forward = self.proportional(
                        area_error,
                        0.000006,
                        float(self.get_parameter('ball_chase_max_speed').value),
                    )
                    if close_area_mode:
                        align_scale = max(0.25, 1.0 - (abs(error_x) / max(center_threshold, 1.0)))
                        base_forward *= align_scale
                    twist.linear.x = forward_sign * base_forward
                    self.last_approach_was_straight = True
                    self.approach_turn_reference_error = None
                    self.approach_turn_stuck_since = None
                    self.approach_turn_stuck_active = False
                elif creep_mode:
                    creep_speed = float(self.get_parameter('ball_close_steer_speed').value) if close_area_mode else float(self.get_parameter('ball_chase_crawl_speed').value)
                    turn_limit = float(self.get_parameter('ball_close_max_turn_speed').value) if close_area_mode else float(self.get_parameter('ball_chase_max_turn_speed').value)
                    twist.linear.x = forward_sign * creep_speed
                    twist.angular.z = self.biased_turn(
                        error_x,
                        float(self.get_parameter('ball_chase_turn_gain').value),
                        turn_limit,
                        float(self.get_parameter('min_chase_turn_speed').value),
                    )
                    twist.angular.z *= turn_sign
                    self.last_approach_was_straight = False
                    self.approach_turn_reference_error = None
                    self.approach_turn_stuck_since = None
                    self.approach_turn_stuck_active = False
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = self.biased_turn(
                        error_x,
                        float(self.get_parameter('ball_chase_turn_gain').value),
                        float(self.get_parameter('ball_close_max_turn_speed').value) if close_area_mode else float(self.get_parameter('ball_chase_max_turn_speed').value),
                        float(self.get_parameter('min_chase_turn_speed').value),
                    )
                    twist.angular.z *= turn_sign
                    self.last_approach_was_straight = False

                    stuck_delta = float(self.get_parameter('approach_turn_stuck_err_delta_px').value)
                    stuck_hold = float(self.get_parameter('approach_turn_stuck_hold_sec').value)
                    if (
                        self.approach_turn_reference_error is None
                        or (self.approach_turn_reference_error * error_x) < 0.0
                        or abs(error_x - self.approach_turn_reference_error) > stuck_delta
                    ):
                        self.approach_turn_reference_error = error_x
                        self.approach_turn_stuck_since = now
                        self.approach_turn_stuck_active = False
                    elif self.approach_turn_stuck_since is not None and (now - self.approach_turn_stuck_since) >= stuck_hold:
                        self.approach_turn_stuck_active = True

                if (
                    self.latest_status.possession_candidate
                    and capture_aligned
                    and abs(twist.angular.z) <= possession_max_turn_cmd
                ):
                    if self.candidate_stable_since is None:
                        self.candidate_stable_since = now
                    elif (now - self.candidate_stable_since) >= float(self.get_parameter('possession_candidate_hold_sec').value):
                        self.last_possession_candidate_time = now
                else:
                    self.candidate_stable_since = None

                debug_message = (
                    f"APPROACH visible=1 err_x={error_x:.1f} err_y={error_y:.1f} area={tracking_area:.0f} "
                    f"close={int(close_area_mode)} thr={center_threshold:.0f} centered={int(centered_enough)} cand={int(self.latest_status.possession_candidate)} "
                    f"cand_stable={self.candidate_stable_since is not None} "
                    f"armed={int(self.last_possession_candidate_time > 0.0 and (now - self.last_possession_candidate_time) <= float(self.get_parameter('blind_zone_capture_timeout_sec').value))} "
                    f"straight={int(self.last_approach_was_straight)} creep={int(creep_mode)} stuck={int(self.approach_turn_stuck_active)} vx={twist.linear.x:.3f} wz={twist.angular.z:.3f}"
                )
            else:
                candidate_recent = (
                    self.last_possession_candidate_time > 0.0 and
                    (now - self.last_possession_candidate_time) <= float(self.get_parameter('blind_zone_capture_timeout_sec').value)
                )
                if candidate_recent and self.last_approach_was_straight:
                    if self.ball_blind_zone_since is None:
                        self.ball_blind_zone_since = now
                    debug_message = (
                        f"APPROACH blind-zone capture candidate_recent=1 straight=1 "
                        f"dt={(now - self.last_possession_candidate_time):.2f} -> BALL_POSSESSION"
                    )
                    self.transition(self.BALL_POSSESSION)
                else:
                    debug_message = (
                        f"APPROACH lost visible=0 candidate_recent={int(candidate_recent)} "
                        f"straight={int(self.last_approach_was_straight)} -> RECOVER"
                    )
                    self.transition(self.RECOVER)

            if debug_message is not None and now >= getattr(self, 'next_approach_debug_time', 0.0):
                self.get_logger().info(debug_message)
                self.next_approach_debug_time = now + 0.25

        elif self.state == self.BALL_POSSESSION:
            self.publish_target('ball')
            self.publish_mode('HOLD')
            self.publish_rgb(0, 255, 0)
            # Demo behavior: once possession is declared, pin the ball for a
            # brief settle window, then stop and stay latched in possession.
            if state_elapsed < float(self.get_parameter('ball_possession_settle_sec').value):
                twist.linear.x = forward_sign * float(self.get_parameter('ball_possession_settle_speed').value)
            else:
                twist.linear.x = 0.0
            self.ball_possession_release_since = None

        elif self.state == self.SEARCH_GOAL:
            self.transition(self.RECOVER)

        elif self.state == self.ALIGN_TO_GOAL:
            self.transition(self.RECOVER)

        elif self.state == self.DRIVE_TO_GOAL:
            self.transition(self.RECOVER)

        elif self.state == self.KICK:
            self.transition(self.RECOVER)

        elif self.state == self.RECOVER:
            self.publish_target('ball')
            self.publish_mode('HOLD')
            self.publish_rgb(255, 255, 0)
            if state_elapsed >= float(self.get_parameter('recover_duration_sec').value):
                self.transition(self.SEARCH_BALL)

        scaled_linear_x = twist.linear.x * float(self.get_parameter('linear_speed_scale').value)
        scaled_angular_z = twist.angular.z * float(self.get_parameter('angular_speed_scale').value)
        min_linear = float(self.get_parameter('min_effective_linear_speed').value)
        min_turn = float(self.get_parameter('min_effective_turn_speed').value)
        if 0.0 < abs(scaled_linear_x) < min_linear:
            scaled_linear_x = min_linear if scaled_linear_x > 0.0 else -min_linear
        if 0.0 < abs(scaled_angular_z) < min_turn:
            scaled_angular_z = min_turn if scaled_angular_z > 0.0 else -min_turn

        if abs(scaled_linear_x) < 1e-6:
            twist.linear.x = 0.0
            self.linear_active_since = None
        else:
            ramped_linear_x = self.limit_rate(
                scaled_linear_x,
                self.last_linear_x,
                float(self.get_parameter('max_linear_step').value),
            )
            twist.linear.x = self.enforce_axis_motion_profile(
                ramped_linear_x,
                now,
                float(self.get_parameter('linear_breakaway_speed').value),
                float(self.get_parameter('linear_hold_speed').value),
                'linear_active_since',
            )

        if abs(scaled_angular_z) < 1e-6:
            twist.angular.z = 0.0
            self.angular_active_since = None
        else:
            ramped_angular_z = self.limit_rate(
                scaled_angular_z,
                self.last_angular_z,
                float(self.get_parameter('max_angular_step').value),
            )
            if self.state == self.SEARCH_BALL:
                twist.angular.z = self.enforce_axis_motion_profile(
                    ramped_angular_z,
                    now,
                    float(self.get_parameter('angular_breakaway_speed').value),
                    float(self.get_parameter('angular_hold_speed').value),
                    'angular_active_since',
                )
            elif self.state == self.APPROACH_BALL and abs(twist.linear.x) < 1e-6:
                breakaway_speed = float(self.get_parameter('approach_turn_breakaway_speed').value)
                hold_speed = float(self.get_parameter('approach_turn_hold_speed').value)
                if self.approach_turn_stuck_active:
                    breakaway_speed = float(self.get_parameter('approach_turn_stuck_breakaway_speed').value)
                    hold_speed = float(self.get_parameter('approach_turn_stuck_hold_speed').value)
                twist.angular.z = self.enforce_axis_motion_profile(
                    ramped_angular_z,
                    now,
                    breakaway_speed,
                    hold_speed,
                    'angular_active_since',
                )
            else:
                hold_floor = float(self.get_parameter('chase_angular_hold_speed').value)
                magnitude = max(abs(ramped_angular_z), hold_floor)
                twist.angular.z = magnitude if ramped_angular_z > 0.0 else -magnitude

        self.last_linear_x = twist.linear.x
        self.last_angular_z = twist.angular.z
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = SoccerFSMNode()
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





