from pathlib import Path
import math

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from capstone_interfaces.msg import SoccerDetection, SoccerDetections
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from ultralytics import YOLO


CLASS_MAP = {
    0: 'ball',
    1: 'goal',
    2: 'turbopi',
}


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('imgsz', 512)
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('publish_topic', '/soccer/detections')
        self.declare_parameter('debug_image_topic', '/soccer/debug/annotated/compressed')
        self.declare_parameter('publish_debug_image', True)
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
        self.declare_parameter('ball_min_fill_ratio', 0.18)

        model_root = Path(get_package_share_directory('capstone_brain')) / 'models' / 'turbopi_ncnn_model'
        self.model = YOLO(str(model_root), task='segment')
        self.bridge = CvBridge()
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.publisher = self.create_publisher(
            SoccerDetections,
            self.get_parameter('publish_topic').value,
            10,
        )
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_image_publisher = None
        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(
                CompressedImage,
                self.get_parameter('debug_image_topic').value,
                10,
            )
        self.latest_frame = None
        self.received_image_count = 0
        self.image_timeout_warned = False
        self.start_time_sec = self.get_clock().now().nanoseconds / 1e9
        self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.last_primary_by_class = {}
        self.debug_image_publish_count = 0
        self.debug_image_encode_warned = False

        self.timer = self.create_timer(0.05, self.process_frame)
        self.get_logger().info(f'Detector node started with model at {model_root}')
        self.get_logger().info(f"Subscribing to camera frames on {self.get_parameter('image_topic').value}")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.received_image_count += 1
        if self.received_image_count == 1:
            self.get_logger().info(f"Receiving camera frames on {self.get_parameter('image_topic').value}")

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

        if candidate.get('fill_ratio', 1.0) < float(self.get_parameter('ball_min_fill_ratio').value):
            return 0.0

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
        if self.latest_frame is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if not self.image_timeout_warned and (now_sec - self.start_time_sec) >= 2.0:
                self.image_timeout_warned = True
                self.get_logger().warning(
                    f"No camera frames received on {self.get_parameter('image_topic').value}"
                )
            return
        frame = self.latest_frame.copy()

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
        masks_xy = []
        if results and getattr(results[0], 'masks', None) is not None and getattr(results[0].masks, 'xy', None) is not None:
            masks_xy = results[0].masks.xy
        if results and len(results[0].boxes) > 0:
            for index, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                class_name = CLASS_MAP.get(cls_id)
                if not class_name:
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

                if class_name == 'ball' and index < len(masks_xy):
                    polygon = masks_xy[index]
                    if polygon is not None and len(polygon) >= 3:
                        polygon = polygon.astype('float32')
                        mask_area = abs(float(cv2.contourArea(polygon)))
                        if mask_area > 1.0:
                            moments = cv2.moments(polygon)
                            if abs(moments['m00']) > 1e-6:
                                center_x = float(moments['m10'] / moments['m00'])
                                center_y = float(moments['m01'] / moments['m00'])
                            bottom_y = float(polygon[:, 1].max())
                            x_coords = polygon[:, 0]
                            y_coords = polygon[:, 1]
                            width = max(1.0, float(x_coords.max() - x_coords.min()))
                            height = max(1.0, float(y_coords.max() - y_coords.min()))
                            area = mask_area
                            fill_ratio = mask_area / max(bbox_area, 1.0)

                candidate = {
                    'class_name': class_name,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bottom_y': bottom_y,
                    'width': width,
                    'height': height,
                    'area': area,
                    'bbox_area': bbox_area,
                    'fill_ratio': fill_ratio,
                    'polygon': None if class_name != 'ball' or index >= len(masks_xy) or masks_xy[index] is None else masks_xy[index].astype('float32'),
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
            item.bbox_bottom_y = detection['bottom_y']
            item.is_primary = True
            msg.detections.append(item)
            new_primary_by_class[detection['class_name']] = detection

        self.last_primary_by_class = new_primary_by_class

        primary_ball = primary_by_class.get('ball')
        if primary_ball is not None:
            polygon = primary_ball.get('polygon')
            label_x = int(round(primary_ball['center_x'] - primary_ball['width'] / 2.0))
            label_y = int(round(primary_ball['center_y'] - primary_ball['height'] / 2.0))
            if polygon is not None and len(polygon) >= 3:
                cv2.polylines(
                    annotated_frame,
                    [polygon.astype('int32').reshape((-1, 1, 2))],
                    True,
                    (0, 255, 0),
                    3,
                )
                label_x = int(round(polygon[:, 0].min()))
                label_y = int(round(polygon[:, 1].min()))
            else:
                x1 = int(round(primary_ball['center_x'] - primary_ball['width'] / 2.0))
                y1 = int(round(primary_ball['center_y'] - primary_ball['height'] / 2.0))
                x2 = int(round(primary_ball['center_x'] + primary_ball['width'] / 2.0))
                y2 = int(round(primary_ball['center_y'] + primary_ball['height'] / 2.0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                annotated_frame,
                'PRIMARY BALL',
                (label_x, max(20, label_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                annotated_frame,
                (int(round(primary_ball['center_x'])), int(round(primary_ball['center_y']))),
                4,
                (255, 255, 255),
                -1,
            )
            cv2.circle(
                annotated_frame,
                (int(round(primary_ball['center_x'])), int(round(primary_ball['bottom_y']))),
                5,
                (0, 0, 255),
                -1,
            )

        self.publisher.publish(msg)
        if self.debug_image_publisher is not None:
            success_encode, encoded = cv2.imencode('.jpg', annotated_frame)
            if success_encode:
                image_msg = CompressedImage()
                image_msg.header.stamp = msg.stamp
                image_msg.format = 'jpeg'
                image_msg.data = encoded.tobytes()
                self.debug_image_publisher.publish(image_msg)
                self.debug_image_publish_count += 1
                if self.debug_image_publish_count == 1:
                    self.get_logger().info(
                        f"Publishing detector debug frames on {self.get_parameter('debug_image_topic').value}"
                    )
            elif not self.debug_image_encode_warned:
                self.debug_image_encode_warned = True
                self.get_logger().warning('Failed to JPEG-encode detector debug frame.')
        if self.show_window:
            cv2.imshow(self.window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Display window requested shutdown with 'q'.")
                if rclpy.ok():
                    rclpy.shutdown()

    def destroy_node(self):
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

