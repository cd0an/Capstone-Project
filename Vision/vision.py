# vision.py
# Tracks balls, goals, and turbopis using a YOLO v8 nano model
# Holds the vision worker function

import cv2
import time
from ultralytics import YOLO
import os

# Classes for trained model
CLASS_MAP = {
    0: "ball",
    1: "goal",
    2: "turbopi"
}

def vision_worker(data_queue):
    # Runs the YOLO model on a dedicated CPU core and pushes 
    # multi-class coordinate payloads to the shared queue
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'turbopi_ncnn_model')
    
    # Load the PyTorch weights
    model = YOLO(model_path, task = 'segment')
    
    cap = cv2.VideoCapture(0)
    
    # Lower resolution for the camera hardware to speed up OpenCV reading 
    # before YOLO resizes it for inference.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[VISION] Camera and YOLO Model loaded successfully.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01) # Prevent CPU thrashing if camera drops a frame
            continue

        results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

        vision_payload = {
            "ball": {"detected": False, "x": 0, "y": 0, "area": 0},
            "goal": {"detected": False, "x": 0, "y": 0, "area": 0},
            "turbopi": {"detected": False, "x": 0, "y": 0, "area": 0}
        }

        if len(results[0].boxes) > 0:
            # Keep track of the largest area seen so far for each class
            # This ensures the robot tracks the closest/biggest target, 
            # ignoring tiny false positives in the background.
            largest_areas = {"ball": 0, "goal": 0, "turbopi": 0}

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_MAP.get(cls_id)
                
                if class_name:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Only update the payload if this is the biggest object of this class in the frame
                    if area > largest_areas[class_name]:
                        largest_areas[class_name] = area
                        
                        vision_payload[class_name]["detected"] = True
                        vision_payload[class_name]["x"] = (x1 + x2) / 2
                        vision_payload[class_name]["y"] = (y1 + y2) / 2
                        vision_payload[class_name]["area"] = area

        # Clear backlog and push the freshest payload
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
            except:
                pass
        
        data_queue.put(vision_payload)