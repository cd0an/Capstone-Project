import csv
import math
import os
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO


CLASS_MAP = {
    0: "ball",
    1: "goal",
    2: "turbopi",
}

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
    width: float
    height: float
    frame_width: int
    frame_height: int
    error_x: float
    error_y: float
    centered: bool
    possession_candidate: bool
    polygon: object


class PlowCalibratorNode(Node):
    def __init__(self):
        super().__init__("plow_calibrator")
        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("imgsz", 512)
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("output_csv", "~/plow_calibration_samples.csv")
        self.declare_parameter("print_period_sec", 1.0)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "TurboPi Plow Calibrator")
        self.declare_parameter("possession_center_tolerance_px", 170.0)
        self.declare_parameter("possession_row_px", 165.0)
        self.declare_parameter("possession_min_area", 4250.0)
        self.declare_parameter("ball_match_distance_px", 220.0)
        self.declare_parameter("ball_track_bonus", 1.75)
        self.declare_parameter("ball_bottom_bias", 0.25)
        self.declare_parameter("ball_center_bias", 0.35)
        self.declare_parameter("ball_edge_margin_px", 50.0)
        self.declare_parameter("ball_edge_penalty", 0.25)
        self.declare_parameter("ball_square_min_ratio", 0.35)
        self.declare_parameter("ball_square_score_floor", 0.25)
        self.declare_parameter("ball_confidence_weight", 0.15)
        self.declare_parameter("ball_min_fill_ratio", 0.18)

        model_root = Path(get_package_share_directory("capstone_brain")) / "models" / "turbopi_ncnn_model"
        self.model = YOLO(str(model_root), task="segment")
        self.bridge = CvBridge()
        self.latest_image = None
        self.received_image_count = 0
        self.image_timeout_warned = False

        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.last_saved_label = ""
        self.last_print_time = 0.0
        self.latest_ball = None
        self.last_primary_ball = None
        self.start_time_sec = self.get_clock().now().nanoseconds / 1e9

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
            keyboard_msg = "stdin is not a TTY; terminal hotkeys disabled"

        self.timer = self.create_timer(0.05, self.control_loop)
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Plow calibrator started with model at {model_root}")
        self.get_logger().info(f"Writing plow calibration samples to {self.output_csv}")
        self.get_logger().info(
            "Hotkeys: o=outside, n=near edge, i=in plow, d=too deep, p=print current sample, q=quit"
        )
        self.get_logger().info(
            f"Subscribing to camera frames on {self.get_parameter('image_topic').value}"
        )
        self.get_logger().info(keyboard_msg)

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.received_image_count += 1
        if self.received_image_count == 1:
            self.get_logger().info(
                f"Receiving camera frames on {self.get_parameter('image_topic').value}"
            )

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
        if os.path.exists(self.output_csv) and os.path.getsize(self.output_csv) > 0:
            return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

    def ball_shape_multiplier(self, width, height):
        if width <= 1e-6 or height <= 1e-6:
            return 0.0
        square_ratio = min(width, height) / max(width, height)
        if square_ratio < float(self.get_parameter("ball_square_min_ratio").value):
            return 0.0
        floor = float(self.get_parameter("ball_square_score_floor").value)
        return floor + (1.0 - floor) * square_ratio

    def candidate_score(self, candidate, frame_width, frame_height):
        score = candidate["area"]
        if candidate.get("fill_ratio", 1.0) < float(self.get_parameter("ball_min_fill_ratio").value):
            return 0.0
        shape_multiplier = self.ball_shape_multiplier(candidate["width"], candidate["height"])
        if shape_multiplier <= 0.0:
            return 0.0
        score *= shape_multiplier

        if self.last_primary_ball is not None:
            distance = math.hypot(
                candidate["center_x"] - self.last_primary_ball["center_x"],
                candidate["center_y"] - self.last_primary_ball["center_y"],
            )
            if distance <= float(self.get_parameter("ball_match_distance_px").value):
                score *= float(self.get_parameter("ball_track_bonus").value)

        vertical_fraction = candidate["center_y"] / max(float(frame_height), 1.0)
        score *= 1.0 + float(self.get_parameter("ball_bottom_bias").value) * vertical_fraction

        frame_center_x = float(frame_width) / 2.0
        horizontal_fraction = min(1.0, abs(candidate["center_x"] - frame_center_x) / max(frame_center_x, 1.0))
        score *= 1.0 + float(self.get_parameter("ball_center_bias").value) * (1.0 - horizontal_fraction)

        edge_margin = float(self.get_parameter("ball_edge_margin_px").value)
        edge_penalty = float(self.get_parameter("ball_edge_penalty").value)
        if candidate["center_x"] <= edge_margin or candidate["center_x"] >= (float(frame_width) - edge_margin):
            score *= edge_penalty

        confidence_weight = float(self.get_parameter("ball_confidence_weight").value)
        score *= 1.0 + confidence_weight * max(0.0, min(1.0, candidate["confidence"]))
        return score

    def detect_primary_ball(self, results, frame_width, frame_height):
        primary_ball = None
        if not results or len(results[0].boxes) == 0:
            self.last_primary_ball = None
            return None

        masks_xy = []
        if getattr(results[0], "masks", None) is not None and getattr(results[0].masks, "xy", None) is not None:
            masks_xy = results[0].masks.xy

        for index, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            class_name = CLASS_MAP.get(cls_id)
            if class_name != "ball":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = float(x2 - x1)
            height = float(y2 - y1)
            bbox_area = float(width * height)
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            bottom_y = float(y2)
            area = bbox_area
            fill_ratio = 1.0
            polygon = None

            if index < len(masks_xy):
                polygon = masks_xy[index]
                if polygon is not None and len(polygon) >= 3:
                    polygon = polygon.astype("float32")
                    mask_area = abs(float(cv2.contourArea(polygon)))
                    if mask_area > 1.0:
                        moments = cv2.moments(polygon)
                        if abs(moments["m00"]) > 1e-6:
                            center_x = float(moments["m10"] / moments["m00"])
                            center_y = float(moments["m01"] / moments["m00"])
                        bottom_y = float(polygon[:, 1].max())
                        x_coords = polygon[:, 0]
                        y_coords = polygon[:, 1]
                        width = max(1.0, float(x_coords.max() - x_coords.min()))
                        height = max(1.0, float(y_coords.max() - y_coords.min()))
                        area = mask_area
                        fill_ratio = mask_area / max(bbox_area, 1.0)

            candidate = {
                "center_x": center_x,
                "center_y": center_y,
                "bottom_y": bottom_y,
                "width": width,
                "height": height,
                "area": area,
                "bbox_area": bbox_area,
                "fill_ratio": fill_ratio,
                "polygon": polygon,
                "confidence": confidence,
            }
            score = self.candidate_score(candidate, frame_width, frame_height)
            if score <= 0.0:
                continue
            candidate["selection_score"] = score
            if primary_ball is None or score > primary_ball["selection_score"]:
                primary_ball = candidate

        self.last_primary_ball = primary_ball
        if primary_ball is None:
            return None

        frame_center_x = float(frame_width) / 2.0
        frame_center_y = float(frame_height) / 2.0
        error_x = primary_ball["center_x"] - frame_center_x
        error_y = primary_ball["center_y"] - frame_center_y
        centered = abs(error_x) <= float(self.get_parameter("possession_center_tolerance_px").value)
        possession_candidate = (
            centered
            and primary_ball["bottom_y"] >= float(self.get_parameter("possession_row_px").value)
            and primary_ball["area"] >= float(self.get_parameter("possession_min_area").value)
        )
        return BallSnapshot(
            center_x=primary_ball["center_x"],
            center_y=primary_ball["center_y"],
            bbox_bottom_y=primary_ball["bottom_y"],
            area=primary_ball["area"],
            confidence=primary_ball["confidence"],
            width=primary_ball["width"],
            height=primary_ball["height"],
            frame_width=frame_width,
            frame_height=frame_height,
            error_x=error_x,
            error_y=error_y,
            centered=centered,
            possession_candidate=possession_candidate,
            polygon=primary_ball["polygon"],
        )

    def current_sample(self, label=""):
        ball = self.latest_ball
        frame_width = int(self.get_parameter("frame_width").value)
        frame_height = int(self.get_parameter("frame_height").value)
        if ball is None:
            return [
                f"{self.now_seconds():.3f}",
                label,
                0,
                0,
                0,
                0,
                0,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                frame_width,
                frame_height,
            ]
        return [
            f"{self.now_seconds():.3f}",
            label,
            1,
            int(ball.possession_candidate),
            int(ball.centered),
            1,
            0,
            f"{ball.error_x:.1f}",
            f"{ball.error_y:.1f}",
            f"{ball.area:.1f}",
            f"{ball.confidence:.3f}",
            f"{ball.bbox_bottom_y:.1f}",
            f"{ball.center_x:.1f}",
            f"{ball.center_y:.1f}",
            ball.frame_width,
            ball.frame_height,
        ]

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def write_sample(self, label):
        row = self.current_sample(label)
        with open(self.output_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
        self.last_saved_label = label
        self.get_logger().info(
            f"saved label={label} visible={row[2]} cand={row[3]} err_x={row[7]} err_y={row[8]} area={row[9]} bottom_y={row[11]}"
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

    def draw_overlay(self, annotated_frame):
        frame_height, frame_width = annotated_frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        tolerance = int(round(float(self.get_parameter("possession_center_tolerance_px").value)))
        plow_row = int(round(float(self.get_parameter("possession_row_px").value)))

        cv2.drawMarker(
            annotated_frame,
            (center_x, center_y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.line(annotated_frame, (0, plow_row), (frame_width - 1, plow_row), (255, 180, 0), 2)
        cv2.line(
            annotated_frame,
            (max(0, center_x - tolerance), 0),
            (max(0, center_x - tolerance), frame_height - 1),
            (255, 180, 0),
            1,
        )
        cv2.line(
            annotated_frame,
            (min(frame_width - 1, center_x + tolerance), 0),
            (min(frame_width - 1, center_x + tolerance), frame_height - 1),
            (255, 180, 0),
            1,
        )

        if self.latest_ball is not None:
            ball = self.latest_ball
            color = (0, 255, 0) if ball.possession_candidate else (0, 220, 255)
            label_x = int(round(ball.center_x - ball.width / 2.0))
            label_y = int(round(ball.center_y - ball.height / 2.0))
            if ball.polygon is not None and len(ball.polygon) >= 3:
                cv2.polylines(
                    annotated_frame,
                    [ball.polygon.astype("int32").reshape((-1, 1, 2))],
                    True,
                    color,
                    3,
                )
                label_x = int(round(ball.polygon[:, 0].min()))
                label_y = int(round(ball.polygon[:, 1].min()))
            else:
                x1 = int(round(ball.center_x - ball.width / 2.0))
                y1 = int(round(ball.center_y - ball.height / 2.0))
                x2 = int(round(ball.center_x + ball.width / 2.0))
                y2 = int(round(ball.center_y + ball.height / 2.0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                annotated_frame,
                "PRIMARY BALL",
                (label_x, max(20, label_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(annotated_frame, (int(round(ball.center_x)), int(round(ball.center_y))), 4, (255, 255, 255), -1)
            cv2.circle(
                annotated_frame,
                (int(round(ball.center_x)), int(round(ball.bbox_bottom_y))),
                5,
                (0, 0, 255),
                -1,
            )

        text_lines = [
            "Hotkeys: o outside | n near_edge | i in_plow | d too_deep | p print | q quit",
            "This tool owns the camera and shows all classes from YOLO.",
            f"last_saved={self.last_saved_label or '-'} row={plow_row} tol={tolerance}",
        ]
        if self.latest_ball is None:
            text_lines.append("primary_ball=none")
        else:
            ball = self.latest_ball
            text_lines.extend(
                [
                    f"cand={int(ball.possession_candidate)} centered={int(ball.centered)} err_x={ball.error_x:.1f} err_y={ball.error_y:.1f}",
                    f"area={ball.area:.1f} conf={ball.confidence:.3f} bottom_y={ball.bbox_bottom_y:.1f}",
                ]
            )
        for index, line in enumerate(text_lines):
            cv2.putText(
                annotated_frame,
                line,
                (10, 24 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    def control_loop(self):
        self.poll_keyboard()
        now = self.now_seconds()
        if self.latest_image is None:
            if not self.image_timeout_warned and (now - self.start_time_sec) >= 2.0:
                self.image_timeout_warned = True
                self.get_logger().warning(
                    f"No camera frames received on {self.get_parameter('image_topic').value}"
                )
            return
        frame = self.latest_image.copy()

        results = self.model.predict(
            frame,
            imgsz=int(self.get_parameter("imgsz").value),
            conf=float(self.get_parameter("confidence_threshold").value),
            verbose=False,
        )

        annotated_frame = frame.copy()
        if results:
            annotated_frame = results[0].plot()

        frame_height, frame_width = frame.shape[:2]
        self.latest_ball = self.detect_primary_ball(results, frame_width, frame_height)
        self.draw_overlay(annotated_frame)

        if self.show_window:
            cv2.imshow(self.window_name, annotated_frame)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code != 255:
                self.handle_key(chr(key_code).lower())

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
