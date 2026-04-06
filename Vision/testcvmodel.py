import cv2
from ultralytics import YOLO

# 1. Load model and explicitly define task='detect' to fix the warning
model = YOLO('turbopi_ncnn_model', task='segment')

# Open the default webcam
cap = cv2.VideoCapture(0)

print("Starting live feed... Press 'q' to stop.")

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Failed to grab a frame from the camera.")
        break

    # Run Inference
    results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

    # Draw the polygons and bounding boxes on the frame
    # This will no longer crash because model.names is explicitly defined
    annotated_frame = results[0].plot()

    # Display the live video feed
    cv2.imshow("TurboPi Live Vision Test", annotated_frame)

    # Wait 1 millisecond for user input; if 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the window when done
cap.release()
cv2.destroyAllWindows()