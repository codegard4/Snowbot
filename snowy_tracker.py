
# Script to provide functionality to detect and track objects in a live video frame

import sys
import cv2
import numpy as np

if __name__ == "__main__":

    # params
    threshold = 0.25 # how confident do we have to be to label an object?
    video = cv2.VideoCapture("vid/3334.mp4") #what input video will we be tracking and detecting objects with?

    # load the model from the models folder
    net = cv2.dnn.readNetFromTensorflow("models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb", "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",)
    tracker = cv2.TrackerGOTURN.create()


    # check if the video opened successfully
    if not video.isOpened():
        print("Whoopsies, the video path is wrong; we could not open it")
        sys.exit()

    classFile = "coco_class_labels.txt"
    with open(classFile) as fp:
        labels = fp.read().split("\n")

    # read and display video frame by frame
    #TODO: add a "Exit" button to the frame instead of just exiting with "q" pressed
    while True:
        ok, frame = video.read()
        if not ok:
            break  # break loop if no more frames

        # prepare each frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        height, width = frame.shape[:2]

        # do we have any detected objects? display them on the screen!
        for i in range(detections.shape[2]):

            # get the confidence of our detected object
            confidence = detections[0, 0, i, 2]

            #if we have high enough confidence display the object on the screen
            if confidence > threshold:

                # get the index of the detection (ex: Person is ID=1)
                class_id = int(detections[0, 0, i, 1])

                # outline the box based on where the object is detected
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x_max, y_max) = box.astype("int")

                # draw the box and label that says what the object is
                label = f"ID {labels[class_id]}: {confidence * 100:.1f}%"
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # display the frame
        cv2.imshow("Snowbot Object Detection and Tracking", frame)

    #TODO: figure out how to add a pause button?

        # break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    video.release()
    cv2.destroyAllWindows()
