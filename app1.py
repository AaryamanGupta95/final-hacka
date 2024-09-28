import cv2
import numpy as np
import threading

# Global variables
latest_result = []
lock = threading.Lock()

# Initialize YOLO model configuration
yolo_weights = "yolov4.weights"  # Path to YOLO weights file
yolo_cfg = "yolov4.cfg"          # Path to YOLO configuration file
coco_names = "coco.names"        # Path to the COCO names file

# Load YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def send_alert(message):
    """Send an alert message (for now just print it)."""
    print(message)  # Replace with desired alert mechanism

def capture_and_detect(camera_index=0):
    global latest_result

    # Try to open camera or video file
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera or video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            # Check if the frame is None (could happen if camera feed is interrupted)
            if frame is None:
                print("Error: Received an empty frame.")
                continue

            height, width, channels = frame.shape
            
            # Prepare the image for the model
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Confidence threshold
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            result = []
            phone_detected = False

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
                    if label == "cell phone":
                        phone_detected = True

            with lock:
                latest_result = result

            if phone_detected:
                send_alert("Alert: Cell phone detected!")

            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # Set color to red for cell phone, green for others
                if label == "cell phone":
                    color = (0, 0, 255)  # Red for cell phone
                else:
                    color = (0, 255, 0)  # Green for other objects

                # Draw the rectangle with the determined color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Live Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"An error occurred during detection: {e}")

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    capture_and_detect(0)  # Use camera index 0
