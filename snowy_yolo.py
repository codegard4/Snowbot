import sys
import cv2
import numpy as np
if __name__ == '__main__':
    # Load YOLOv8 ONNX model
    model_path = "models/yolov8n.onnx"  # Ensure you have a YOLOv8 ONNX model
    net = cv2.dnn.readNetFromONNX(model_path)

    # Set parameters
    threshold = 0.25  # Confidence threshold
    video = cv2.VideoCapture("vid/3334.mp4")  # Load input video

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Could not open video file.")
        sys.exit()

    # Load COCO class labels
    # class_file = "coco_class_labels.txt"
    # with open(class_file, "r") as f:
    #     labels = f.read().strip().split("\n")

    paused = False  # Pause state

    while True:
        if not paused:
            ok, frame = video.read()
            if not ok:
                break  # Exit loop if no more frames

            height, width = frame.shape[:2]

            # Prepare the frame for YOLOv8 model
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            # Process YOLOv8 detections
            for detection in detections[0]:  # YOLOv8 outputs all detections in a single array
                confidence = detection[4]  # Object confidence score
                if confidence > threshold:
                    class_scores = detection[5:]  # Class probabilities
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]

                    if class_confidence > threshold:
                        # Extract bounding box coordinates
                        x, y, w, h = detection[0:4] * np.array([width, height, width, height])
                        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

                        # Draw bounding box and label
                        # label = f"{labels[class_id]}: {class_confidence * 100:.1f}%"
                        label = f"{class_confidence * 100:.1f}%"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("YOLOv8 Object Detection", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/unpause
            paused = not paused

    # Release resources
    video.release()
    cv2.destroyAllWindows()
