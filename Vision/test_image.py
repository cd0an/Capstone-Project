import cv2
import json
from ultralytics import YOLO

CLASS_MAP = {
    0: "ball",
    1: "goal",
    2: "turbopi"
}

def test_single_image(image_path, model_path='best.pt'):
    # Load the model
    model = YOLO(model_path)
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image at {image_path}. Check the file path.")
        return

    print(f"Running inference on {image_path}...")
    
    # Run prediction
    results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

    # Initialize a clean payload
    vision_payload = {
        "ball": {"detected": False, "x": 0, "y": 0, "area": 0},
        "goal": {"detected": False, "x": 0, "y": 0, "area": 0},
        "turbopi": {"detected": False, "x": 0, "y": 0, "area": 0}
    }

    # Extract data and populate the payload
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_MAP.get(cls_id)
            
            if class_name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                vision_payload[class_name]["detected"] = True
                # Rounding the numbers just to make the printout look cleaner
                vision_payload[class_name]["x"] = round((x1 + x2) / 2, 2)
                vision_payload[class_name]["y"] = round((y1 + y2) / 2, 2)
                vision_payload[class_name]["area"] = round((x2 - x1) * (y2 - y1), 2)

    # Print the dictionary to the terminal
    print("\n--- GENERATED VISION PAYLOAD ---")
    print(json.dumps(vision_payload, indent=4))
    print("--------------------------------\n")
    
    # Show the annotated image to visually verify the payload data
    annotated_frame = results[0].plot()
    cv2.imshow("Detection Verification", annotated_frame)
    print("Press any key in the image window to close it.")
    
    cv2.waitKey(0) # Pauses the script until you press a key
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TEST_IMAGE_FILE = "test_photo.jpg" 
    
    test_single_image(TEST_IMAGE_FILE)