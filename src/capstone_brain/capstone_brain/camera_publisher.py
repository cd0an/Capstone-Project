import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__("camera_publisher")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("publish_topic", "image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("period_sec", 0.05)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image,
            str(self.get_parameter("publish_topic").value),
            qos_profile_sensor_data,
        )

        camera_index = int(self.get_parameter("camera_index").value)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("frame_width").value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("frame_height").value))

        self.camera_error_warned = False
        self.timer = self.create_timer(float(self.get_parameter("period_sec").value), self.process_frame)
        self.get_logger().info(
            f"Publishing camera frames from /dev/video{camera_index} on {self.get_parameter('publish_topic').value}"
        )

    def process_frame(self):
        if not self.cap.isOpened():
            if not self.camera_error_warned:
                self.camera_error_warned = True
                self.get_logger().error("Camera is not available on /dev/video0.")
            return

        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning("Camera frame read failed.")
            return

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = str(self.get_parameter("frame_id").value)
        self.publisher.publish(image_msg)

    def destroy_node(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
