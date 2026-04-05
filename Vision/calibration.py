# calibration.py
# Provides an interactive tool to calibrate HSV color
# thresholds using trackbars and saves the settings to config.json

import cv2
import numpy as np
import json
import os

def nothing(x):
    pass

# Setup Windows and Trackbars
cv2.namedWindow("Calibration")
cv2.createTrackbar("L-H", "Calibration", 0, 179, nothing)
cv2.createTrackbar("L-S", "Calibration", 0, 255, nothing)
cv2.createTrackbar("L-V", "Calibration", 0, 255, nothing)
cv2.createTrackbar("U-H", "Calibration", 179, 179, nothing)
cv2.createTrackbar("U-S", "Calibration", 255, 255, nothing)
cv2.createTrackbar("U-V", "Calibration", 255, 255, nothing)

cap = cv2.VideoCapture(0)

print("Adjust sliders. Press 's' to Save and Start Tracking, or 'q' to Quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Read trackbar positions
    values = {
        "low": [cv2.getTrackbarPos("L-H", "Calibration"), 
                cv2.getTrackbarPos("L-S", "Calibration"), 
                cv2.getTrackbarPos("L-V", "Calibration")],
        "high": [cv2.getTrackbarPos("U-H", "Calibration"), 
                cv2.getTrackbarPos("U-S", "Calibration"), 
                cv2.getTrackbarPos("U-V", "Calibration")]
    }

    mask = cv2.inRange(hsv, np.array(values["low"]), np.array(values["high"]))
    cv2.imshow("Calibration", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save to JSON
        with open("config.json", "w") as f:
            json.dump(values, f)
        print("Config saved! Launching tracker...")
        break
    elif key == ord('q'):
        exit()

cap.release()
cv2.destroyAllWindows()

# Automatically run the tracking script
os.system("python ball_tracking.py")