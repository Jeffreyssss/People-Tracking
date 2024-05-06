import torch
import cv2

class Detector:
    def __init__(self):
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Set model to inference mode
        self.model = self.model.eval()

    def detect(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform object detection on the frame
        results = self.model(frame_rgb)

        # Get boxes, labels,confidence scores
        boxes = results.pred[0][:, :4].cpu().numpy()
        labels = results.pred[0][:, -1].cpu().numpy().astype(int)
        confidences = results.pred[0][:, 4].cpu().numpy()  # Confidence scores are typically at index 4

        height, width, _ = frame.shape

        # List to store bounding boxes including class ID and confidence
        person_boxes = []

        # Iterate over the detected objects
        for box, label, confidence in zip(boxes, labels, confidences):
            if label == 0:  # Check if the detected object is a person
                x1, y1, x2, y2 = box.astype(int)
                # Create a tuple with necessary information
                bbox_info = (x1, y1, x2, y2, label, confidence)
                # Append the detailed bounding box to the list
                person_boxes.append(bbox_info)

        return person_boxes








