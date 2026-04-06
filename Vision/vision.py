# vision.py
# Tracks balls, goals, and turbopis using a YOLO v8 nano model
# Holds the vision worker function

import cv2
from ultralytics import YOLO

# Classes for trained model
CLASS_MAP = {
    0: "ball",
    1: "goal",
    2: "turbopi"
}

def vision_worker(data_queue):
    # Runs the YOLO model on a dedicated CPU core and pushes 
    # multi-class coordinate payloads to the shared queue
    model = YOLO('Vision/best.py') 
    
    cap = cv2.VideoCapture(0)
    print("[VISION] Camera and YOLO Model loaded successfully.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

        vision_payload = {
            "ball": {"detected": False, "x": 0, "y": 0, "area": 0},
            "goal": {"detected": False, "x": 0, "y": 0, "area": 0},
            "turbopi": {"detected": False, "x": 0, "y": 0, "area": 0}
        }

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = CLASS_MAP.get(cls_id)
                
                if class_name:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vision_payload[class_name]["detected"] = True
                    vision_payload[class_name]["x"] = (x1 + x2) / 2
                    vision_payload[class_name]["y"] = (y1 + y2) / 2
                    vision_payload[class_name]["area"] = (x2 - x1) * (y2 - y1)

        # Clear backlog and push the freshest payload
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
            except:
                pass
        
        data_queue.put(vision_payload)