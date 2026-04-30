import csv
import os
import select
import sys
import termios
import tty
from dataclasses import dataclass

import rclpy
from capstone_interfaces.msg import SoccerDetections, TrackingStatus
from rclpy.node import Node


LABELS = {
    "o": "outside",
    "n": "near_edge",
    "i": "in_plow",
    "d": "too_deep",
}


@dataclass
class BallSnapshot:
    center_x: float
    center_y: float
    bbox_bottom_y: float
    area: float
    confidence: float
    frame_width: int
    frame_height: int


class PlowCalibratorNode(Node):
    def __init__(self):
        super().__init__("plow_calibrator")
        self.declare_parameter("detection_topic", "/soccer/detections")
        self.declare_parameter("status_topic", "/soccer/tracking_status")
        self.declare_parameter("output_csv", "~/plow_calibration_samples.csv")
        self.declare_parameter("print_period_sec", 0.25)

        self.latest_status = TrackingStatus()
        self.latest_ball = None
        self.last_print_time = 0.0

        output_csv = os.path.expanduser(str(self.get_parameter("output_csv").value))
        self.output_csv = os.path.abspath(output_csv)
        self.ensure_csv_header()

        self.stdin_fd = None
        self.stdin_settings = None
        if sys.stdin.isatty():
            self.stdin_fd = sys.stdin.fileno()
            self.stdin_settings = termios.tcgetattr(self.stdin_fd)
            tty.setcbreak(self.stdin_fd)
            keyboard_msg = "keyboard capture enabled"
        else:
            keyboard_msg = "stdin is not a TTY; label hotkeys disabled"

        self.create_subscription(
            SoccerDetections,
            str(self.get_parameter("detection_topic").value),
            self.detections_callback,
            10,
        )
        self.create_subscription(
            TrackingStatus,
            str(self.get_parameter("status_topic").value),
            self.status_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f"Writing plow calibration samples to {self.output_csv}")
        self.get_logger().info(
            "Hotkeys: o=outside, n=near edge, i=in plow, d=too deep, p=print current sample, q=quit"
        )
        self.get_logger().info(keyboard_msg)

    def ensure_csv_header(self):
        header = [
            "timestamp_sec",
            "label",
            "visible",
            "possession_candidate",
            "centered",
            "in_range",
            "stale",
            "error_x",
            "error_y",
            "area",
            "confidence",
            "bbox_bottom_y",
            "center_x",
            "center_y",
            "frame_width",
            "frame_height",
        ]
        file_exists = os.path.exists(self.output_csv)
        if file_exists and os.path.getsize(self.output_csv) > 0:
            return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

    def detections_callback(self, msg):
        self.latest_ball = None
        for detection in msg.detections:
            if detection.class_name == "ball" and detection.is_primary:
                self.latest_ball = BallSnapshot(
                    center_x=float(detection.center_x),
                    center_y=float(detection.center_y),
                    bbox_bottom_y=float(detection.bbox_bottom_y),
                    area=float(detection.area),
                    confidence=float(detection.confidence),
                    frame_width=int(detection.frame_width),
                    frame_height=int(detection.frame_height),
                )
                break

    def status_callback(self, msg):
        self.latest_status = msg

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def current_sample(self, label=""):
        ball = self.latest_ball
        status = self.latest_status
        return [
            f"{self.now_seconds():.3f}",
            label,
            int(status.visible),
            int(status.possession_candidate),
            int(status.centered),
            int(status.in_range),
            int(status.stale),
            f"{float(status.error_x):.1f}",
            f"{float(status.error_y):.1f}",
            f"{float(status.area):.1f}",
            f"{float(status.confidence):.3f}",
            "" if ball is None else f"{ball.bbox_bottom_y:.1f}",
            "" if ball is None else f"{ball.center_x:.1f}",
            "" if ball is None else f"{ball.center_y:.1f}",
            int(status.frame_width),
            int(status.frame_height),
        ]

    def write_sample(self, label):
        row = self.current_sample(label)
        with open(self.output_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
        self.get_logger().info(
            f"saved label={label} err_x={row[7]} err_y={row[8]} area={row[9]} bottom_y={row[11]} cand={row[3]}"
        )

    def print_current_state(self):
        row = self.current_sample()
        self.get_logger().info(
            f"sample visible={row[2]} cand={row[3]} centered={row[4]} err_x={row[7]} err_y={row[8]} area={row[9]} bottom_y={row[11]}"
        )

    def poll_keyboard(self):
        if self.stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self.stdin_fd], [], [], 0.0)
            if not ready:
                break
            key = os.read(self.stdin_fd, 1).decode("utf-8", errors="ignore").lower()
            if key in LABELS:
                self.write_sample(LABELS[key])
            elif key == "p":
                self.print_current_state()
            elif key == "q":
                self.get_logger().info("quit requested from keyboard")
                if rclpy.ok():
                    rclpy.shutdown()
                return

    def control_loop(self):
        self.poll_keyboard()
        now = self.now_seconds()
        if now - self.last_print_time >= float(self.get_parameter("print_period_sec").value):
            self.last_print_time = now
            self.print_current_state()

    def destroy_node(self):
        if self.stdin_fd is not None and self.stdin_settings is not None:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_settings)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlowCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
