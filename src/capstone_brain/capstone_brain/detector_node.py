from pathlib import Path
import math

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from capstone_interfaces.msg import SoccerDetection, SoccerDetections
from rclpy.node import Node
from ultralytics import YOLO


CLASS_MAP = {
    0: 'ball',
    1: 'goal',
    2: 'turbopi',
}


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('imgsz', 512)
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('publish_topic', '/soccer/detections')
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'TurboPi Live Vision')
        self.declare_parameter('ball_match_distance_px', 220.0)
        self.declare_parameter('ball_track_bonus', 1.75)
        self.declare_parameter('ball_bottom_bias', 0.25)
        self.declare_parameter('ball_center_bias', 0.35)
        self.declare_parameter('ball_edge_margin_px', 50.0)
        self.declare_parameter('ball_edge_penalty', 0.25)
        self.declare_parameter('ball_square_min_ratio', 0.35)
        self.declare_parameter('ball_square_score_floor', 0.25)
        self.declare_parameter('ball_confidence_weight', 0.15)

        model_root = Path(get_package_share_directory('capstone_brain')) / 'models' / 'turbopi_ncnn_model'
        self.model = YOLO(str(model_root), task='segment')
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.publisher = self.create_publisher(
            SoccerDetections,
            self.get_parameter('publish_topic').value,
            10,
        )

        camera_index = int(self.get_parameter('camera_index').value)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter('frame_width').value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter('frame_height').value))

        self.last_primary_by_class = {}

        self.timer = self.create_timer(0.05, self.process_frame)
        self.get_logger().info(f'Detector node started with model at {model_root}')

    def ball_shape_multiplier(self, width, height):
        if width <= 1e-6 or height <= 1e-6:
            return 0.0
        square_ratio = min(width, height) / max(width, height)
        if square_ratio < float(self.get_parameter('ball_square_min_ratio').value):
            return 0.0
        floor = float(self.get_parameter('ball_square_score_floor').value)
        return floor + (1.0 - floor) * square_ratio

    def candidate_score(self, class_name, candidate, frame_width, frame_height):
        score = candidate['area']
        if class_name != 'ball':
            return score

        shape_multiplier = self.ball_shape_multiplier(candidate['width'], candidate['height'])
        if shape_multiplier <= 0.0:
            return 0.0
        score *= shape_multiplier

        previous = self.last_primary_by_class.get('ball')
        if previous is not None:
            distance = math.hypot(
                candidate['center_x'] - previous['center_x'],
                candidate['center_y'] - previous['center_y'],
            )
            if distance <= float(self.get_parameter('ball_match_distance_px').value):
                score *= float(self.get_parameter('ball_track_bonus').value)

        vertical_fraction = candidate['center_y'] / max(float(frame_height), 1.0)
        score *= 1.0 + float(self.get_parameter('ball_bottom_bias').value) * vertical_fraction

        frame_center_x = float(frame_width) / 2.0
        horizontal_fraction = min(1.0, abs(candidate['center_x'] - frame_center_x) / max(frame_center_x, 1.0))
        score *= 1.0 + float(self.get_parameter('ball_center_bias').value) * (1.0 - horizontal_fraction)

        edge_margin = float(self.get_parameter('ball_edge_margin_px').value)
        edge_penalty = float(self.get_parameter('ball_edge_penalty').value)
        if candidate['center_x'] <= edge_margin or candidate['center_x'] >= (float(frame_width) - edge_margin):
            score *= edge_penalty

        # Confidence should only gently break ties. The model is too noisy to let
        # confidence dominate ball choice near the plow.
        confidence_weight = float(self.get_parameter('ball_confidence_weight').value)
        score *= 1.0 + confidence_weight * max(0.0, min(1.0, candidate['confidence']))
        return score

    def process_frame(self):
        if not self.cap.isOpened():
            self.get_logger().error('Camera is not available on /dev/video0.')
            return

        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning('Camera frame read failed.')
            return

        results = self.model.predict(
            frame,
            imgsz=int(self.get_parameter('imgsz').value),
            conf=float(self.get_parameter('confidence_threshold').value),
            verbose=False,
        )

        annotated_frame = frame.copy()
        if results:
            annotated_frame = results[0].plot()
        frame_height, frame_width = frame.shape[:2]
        cv2.drawMarker(
            annotated_frame,
            (frame_width // 2, frame_height // 2),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

        primary_by_class = {}
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_MAP.get(cls_id)
                if not class_name:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = float(x2 - x1)
                height = float(y2 - y1)
                area = float(width * height)
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                candidate = {
                    'class_name': class_name,
                    'center_x': float((x1 + x2) / 2.0),
                    'center_y': float((y1 + y2) / 2.0),
                    'width': width,
                    'height': height,
                    'area': area,
                    'confidence': confidence,
                }
                candidate_score = self.candidate_score(class_name, candidate, frame_width, frame_height)
                if candidate_score <= 0.0:
                    continue
                if (
                    class_name not in primary_by_class
                    or candidate_score > primary_by_class[class_name]['selection_score']
                ):
                    candidate['selection_score'] = candidate_score
                    primary_by_class[class_name] = candidate

        msg = SoccerDetections()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_width = int(frame_width)
        msg.frame_height = int(frame_height)

        new_primary_by_class = {}
        for detection in primary_by_class.values():
            item = SoccerDetection()
            item.stamp = msg.stamp
            item.class_name = detection['class_name']
            item.center_x = detection['center_x']
            item.center_y = detection['center_y']
            item.area = detection['area']
            item.confidence = detection['confidence']
            item.frame_width = int(frame_width)
            item.frame_height = int(frame_height)
            item.is_primary = True
            msg.detections.append(item)
            new_primary_by_class[detection['class_name']] = detection

        self.last_primary_by_class = new_primary_by_class
        self.publisher.publish(msg)
        if self.show_window:
            cv2.imshow(self.window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Display window requested shutdown with 'q'.")
                if rclpy.ok():
                    rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if getattr(self, 'show_window', False):
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

