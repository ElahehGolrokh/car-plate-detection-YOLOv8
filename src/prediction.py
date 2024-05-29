import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List
from ultralytics import YOLO


class Predict:
    """
    Gets predictions and visualization of a yolo saved model
    ...
    Attributes
    ----------
        image_path: path to jpg or png test image
        model_path: path to saved YOLOv8 model
        output_path: name of plt saved figure of final predictio

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

    def visualize_predictions(self, predictions: list, class_names: List[str]):
        """
        Visualizes YOLO predictions on an image using OpenCV.

        :param predictions: List of predictions
        :param class_names: List of class names.
        """
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, ax = plt.subplots(1)
        ax.imshow(image)

        # Add bounding boxes
        for pred in predictions:
            x_min, y_min, x_max, y_max, confidence, class_id = pred

            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min),
                                     x_max - x_min,
                                     y_max - y_min,
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
        plt.savefig(self.output_path)
