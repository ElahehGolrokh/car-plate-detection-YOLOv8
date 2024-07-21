import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from ultralytics import YOLO


class ImagePredictor:
    """
    Gets predictions and visualization of a yolo saved model on test image
    ...
    Attributes
    ----------
        image_path: path to jpg or png test image
        model_path: path to saved YOLOv8 model
        output_path: name of plt saved figure of final prediction

    Public Methods
        get_yolo_predictions()
        visualize_predictions()
    """
    def __init__(self,
                 image_path: str,
                 model_path: str,
                 output_path: str) -> None:
        self.image_path = image_path
        self.model_path = model_path
        self.output_path = output_path

    def get_yolo_predictions(self) -> list:
        """
        Get YOLOv8 predictions for an image.

        :return: List of predictions
        """
        # Load the model
        model = YOLO(self.model_path)

        # Get predictions
        results = model.predict(self.image_path)

        predictions = []

        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                predictions.append([x_min, y_min, x_max, y_max, confidence, class_id])

        return predictions

    def visualize_predictions(self,
                              predictions: list,
                              class_names: List[str],
                              ocr_result: Tuple):
        """
        Visualizes YOLO predictions on an image using OpenCV.

        :param predictions: List of predictions
        :param class_names: List of class names
        :param ocr_result: Result of plate reading via easyocr
        """
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, ax = plt.subplots(1)
        ax.imshow(image)

        # Add bounding boxes
        for pred in predictions:
            x_min, y_min, x_max, y_max, confidence, class_id = pred
            width = x_max - x_min
            height = y_max - y_min

            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min),
                                     width,
                                     height,
                                     linewidth=2,
                                     edgecolor='g',
                                     facecolor='none')
            ax.add_patch(rect)

            # Add label
            label = f"{class_names[class_id]}: {confidence:.2f}"
            plt.text(x_min,
                     y_min - 10,
                     label,
                     color='g',
                     fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5))
            
            # Add OCR Result
            if ocr_result:
                label = f"Plate Number: {ocr_result[0][1]}, Confidence: {ocr_result[0][2]:.2f}"
            else:
                label = 'Unable to read'
            plt.text(x_min - width/2,
                     y_max + height/2,
                     label,
                     color='black',
                     fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))
        plt.savefig(self.output_path)


class VideoPredictor:
    """
    Gets predictions and visualization of a yolo saved model on test video
    ...
    Attributes
    ----------
        video_path: path to jpg or png test video
        model_path: path to saved YOLOv8 model
        output_path: name of saved avi final of prediction

    Public Methods
        run()
    """
    def __init__(self,
                 video_path: str,
                 model_path: str,
                 output_path: str) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path

    def run(self) -> None:

        model = YOLO(self.model_path)

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path,
                              fourcc,
                              fps,
                              (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on the frame
            results = model(frame)

            # Extract predictions and draw bounding boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                confidence = box.conf.item()
                color = (0, 255, 0)  # Green color for bounding boxes

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw label and confidence
                cv2.putText(frame,
                            f'{label} {confidence:.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2)

            # Write the frame with predictions
            out.write(frame)

            # Display the frame with predictions
            cv2.imshow('YOLOv8 Predictions', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
