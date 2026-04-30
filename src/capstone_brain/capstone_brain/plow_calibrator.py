import csv
import os
import select
import sys
import termios
import tty
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from capstone_interfaces.msg import SoccerDetections, TrackingStatus
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


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
        self.declare_parameter("debug_image_topic", "/soccer/debug/annotated/compressed")
        self.declare_parameter("output_csv", "~/plow_calibration_samples.csv")
        self.declare_parameter("print_period_sec", 1.0)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "TurboPi Plow Calibrator")
        self.declare_parameter("display_width", 640)
        self.declare_parameter("display_height", 480)
        self.declare_parameter("possession_center_tolerance_px", 170.0)
        self.declare_parameter("possession_row_px", 165.0)

        self.latest_status = TrackingStatus()
        self.latest_ball = None
        self.latest_debug_frame = None
        self.last_print_time = 0.0
        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.last_saved_label = ""

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
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("debug_image_topic").value),
            self.debug_image_callback,
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

    def debug_image_callback(self, msg):
        if not msg.data:
            return
        image_array = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is not None:
            self.latest_debug_frame = frame

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
        self.last_saved_label = label
        self.get_logger().info(
            f"saved label={label} err_x={row[7]} err_y={row[8]} area={row[9]} bottom_y={row[11]} cand={row[3]}"
        )

    def print_current_state(self):
        row = self.current_sample()
        self.get_logger().info(
            f"PRINT sample visible={row[2]} cand={row[3]} centered={row[4]} err_x={row[7]} err_y={row[8]} area={row[9]} bottom_y={row[11]}"
        )

    def poll_keyboard(self):
        if self.stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self.stdin_fd], [], [], 0.0)
            if not ready:
                break
            key = os.read(self.stdin_fd, 1).decode("utf-8", errors="ignore").lower()
            self.handle_key(key)

    def handle_key(self, key):
        if key in LABELS:
            self.write_sample(LABELS[key])
        elif key == "p":
            self.print_current_state()
        elif key == "q":
            self.get_logger().info("quit requested from keyboard")
            if rclpy.ok():
                rclpy.shutdown()

    def estimated_bbox(self, ball, frame_width, frame_height):
        if ball is None:
            return None
        half_height = max(1.0, float(ball.bbox_bottom_y) - float(ball.center_y))
        height = max(2.0, 2.0 * half_height)
        width = max(2.0, float(ball.area) / height)
        x1 = int(round(ball.center_x - width / 2.0))
        x2 = int(round(ball.center_x + width / 2.0))
        y1 = int(round(ball.bbox_bottom_y - height))
        y2 = int(round(ball.bbox_bottom_y))
        x1 = max(0, min(frame_width - 1, x1))
        x2 = max(0, min(frame_width - 1, x2))
        y1 = max(0, min(frame_height - 1, y1))
        y2 = max(0, min(frame_height - 1, y2))
        return x1, y1, x2, y2

    def render_display(self):
        if not self.show_window:
            return

        status = self.latest_status
        ball = self.latest_ball
        if self.latest_debug_frame is not None:
            canvas = self.latest_debug_frame.copy()
            frame_height, frame_width = canvas.shape[:2]
        else:
            frame_width = int(status.frame_width) if int(status.frame_width) > 0 else int(self.get_parameter("display_width").value)
            frame_height = int(status.frame_height) if int(status.frame_height) > 0 else int(self.get_parameter("display_height").value)
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        center_x = frame_width // 2
        center_y = frame_height // 2
        tolerance = int(round(float(self.get_parameter("possession_center_tolerance_px").value)))
        plow_row = int(round(float(self.get_parameter("possession_row_px").value)))

        cv2.line(canvas, (center_x, 0), (center_x, frame_height - 1), (80, 80, 80), 1)
        cv2.line(canvas, (0, center_y), (frame_width - 1, center_y), (60, 60, 60), 1)
        cv2.line(canvas, (0, plow_row), (frame_width - 1, plow_row), (255, 180, 0), 2)
        cv2.line(canvas, (max(0, center_x - tolerance), 0), (max(0, center_x - tolerance), frame_height - 1), (255, 180, 0), 1)
        cv2.line(canvas, (min(frame_width - 1, center_x + tolerance), 0), (min(frame_width - 1, center_x + tolerance), frame_height - 1), (255, 180, 0), 1)

        bbox = self.estimated_bbox(ball, frame_width, frame_height)
        if bbox is not None:
            candidate_color = (0, 220, 0) if status.possession_candidate else (0, 180, 255)
            if status.centered and status.possession_candidate:
                candidate_color = (0, 255, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), candidate_color, 2)
            cv2.circle(canvas, (int(round(ball.center_x)), int(round(ball.center_y))), 4, (255, 255, 255), -1)
            cv2.circle(canvas, (int(round(ball.center_x)), int(round(ball.bbox_bottom_y))), 5, (0, 0, 255), -1)

        text_lines = [
            "Hotkeys: o outside | n near_edge | i in_plow | d too_deep | p print | q quit",
            "Green detector box = PRIMARY BALL selected by detector_node",
            f"visible={int(status.visible)} cand={int(status.possession_candidate)} centered={int(status.centered)} in_range={int(status.in_range)} stale={int(status.stale)}",
            f"err_x={float(status.error_x):.1f} err_y={float(status.error_y):.1f} area={float(status.area):.1f} conf={float(status.confidence):.3f}",
            f"bottom_y={'' if ball is None else f'{ball.bbox_bottom_y:.1f}'} center_x={'' if ball is None else f'{ball.center_x:.1f}'} center_y={'' if ball is None else f'{ball.center_y:.1f}'}",
            f"last_saved={self.last_saved_label or '-'} row={plow_row} tol={tolerance}",
        ]
        for index, line in enumerate(text_lines):
            cv2.putText(
                canvas,
                line,
                (10, 24 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, canvas)
        key_code = cv2.waitKey(1) & 0xFF
        if key_code != 255:
            self.handle_key(chr(key_code).lower())

    def control_loop(self):
        self.poll_keyboard()
        self.render_display()
        now = self.now_seconds()
        if now - self.last_print_time >= float(self.get_parameter("print_period_sec").value):
            self.last_print_time = now
            self.print_current_state()

    def destroy_node(self):
        if self.stdin_fd is not None and self.stdin_settings is not None:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_settings)
        if self.show_window:
            cv2.destroyAllWindows()
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
