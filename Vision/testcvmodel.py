import cv2
from ultralytics import YOLO

# Load trained PyTorch model
model = YOLO('/turbopi_ncnn_model/model_ncnn.py')

# Open the default webcam
# If you plugged in an external USB camera, you might need to change this to 1 or 2
cap = cv2.VideoCapture(0)

print("Starting live feed... Press 'q' to stop.")

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Failed to grab a frame from the camera.")
        break

    # Run Inference
    # imgsz=512 ensures the black-edge padding matches your Roboflow dataset
    # conf=0.5 means it will only show polygons if it is 50%+ sure
    # verbose=False stops the terminal from being spammed with text every frame
    results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

    # Draw the polygons and bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the live video feed
    cv2.imshow("TurboPi Live Vision Test", annotated_frame)

    # Wait 1 millisecond for user input; if 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the window when done
cap.release()
cv2.destroyAllWindows()