from flask import Flask, request, jsonify
import cv2
import numpy as np
import threading
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# YOLO model configuration
yolo_weights = "yolov4.weights"
yolo_cfg = "yolov4.cfg"
coco_names = "coco.names"

# Load the YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variable to hold latest detection result
latest_result = {"error": "No detection yet"}
lock = threading.Lock()  # For thread safety when accessing shared variables

def capture_and_detect(camera_index=0):
    """
    Captures a frame from the live camera feed every 10 seconds,
    performs object detection, and updates the global detection result.
    """
    global latest_result

    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture and process frames indefinitely
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection on the frame
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process YOLO detections
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-max Suppression to reduce overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Prepare the result for JSON response
        result = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                box = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                result.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "box": box
                })

        # Update the global latest result
        with lock:
            latest_result = result

        # Display the frame (optional for debugging)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame in a window (only works in the main thread)
        cv2.imshow("Live Camera", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return "YOLO Object Detection API on Live Camera Feed. Use the /latest_detection endpoint to view results."

@app.route('/latest_detection', methods=['GET'])
def latest_detection():
    """
    Returns the latest detection results from the live camera feed.
    """
    with lock:
        return jsonify(latest_result), 200

if __name__ == '__main__':
    # Start the live video capture and object detection in a separate thread
    threading.Thread(target=capture_and_detect, args=(0,)).start()
    app.run(debug=True)
