from pathlib import Path

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
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_topic', '/soccer/detections')

        model_root = Path(get_package_share_directory('capstone_brain')) / 'models' / 'turbopi_ncnn_model'
        self.model = YOLO(str(model_root), task='segment')
        self.publisher = self.create_publisher(
            SoccerDetections,
            self.get_parameter('publish_topic').value,
            10,
        )

        camera_index = int(self.get_parameter('camera_index').value)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter('frame_width').value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter('frame_height').value))

        self.timer = self.create_timer(0.05, self.process_frame)
        self.get_logger().info(f'Detector node started with model at {model_root}')

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

        frame_height, frame_width = frame.shape[:2]
        largest_by_class = {}
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_MAP.get(cls_id)
                if not class_name:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = float((x2 - x1) * (y2 - y1))
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                candidate = {
                    'class_name': class_name,
                    'center_x': float((x1 + x2) / 2.0),
                    'center_y': float((y1 + y2) / 2.0),
                    'area': area,
                    'confidence': confidence,
                }
                if class_name not in largest_by_class or area > largest_by_class[class_name]['area']:
                    largest_by_class[class_name] = candidate

        msg = SoccerDetections()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_width = int(frame_width)
        msg.frame_height = int(frame_height)

        for detection in largest_by_class.values():
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

        self.publisher.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
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
        rclpy.shutdown()
