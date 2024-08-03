import easyocr
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Tuple
from ultralytics import YOLO


class PrecictorBase(ABC):

    """
    Gets predictions and visualization of a yolo saved model on test images
    and videos

    Notes:
        - you have to override these methods: ``_get_yolo_predictions``,
                                              ``_visualize_predictions``,
                                              ``run()``
        - don't override these methods: ``_check_input``,
                                        ``_crop_plate``,
                                        ``_read_plate``

    ...
    Attributes
    ----------
        input: path to input image/video or dir of test images
        model_path: path to saved YOLOv8 model
        output_name: name of plt saved figure of final prediction
        reader: easyocr.Reader which would be passed if the user wants to
                read the plate number

    Private Methods
    ---------------
        _check_input()
        _get_yolo_predictions()
        _visualize_predictions()
        _crop_plate()
        _read_plate()

    Public Methods
    --------------
        run()

    Examples:
        >>> predictor = PrecictorBase(input,
                                      model_path,
                                      output_name,
                                      reader)
        >>> predictor.run()

    """
    def __init__(self,
                 input: str,
                 model_path: str,
                 output_name: str,
                 reader: easyocr.Reader = None) -> None:
        self.input = input
        self.model = YOLO(model_path)
        self.output_name = output_name
        self.reader = reader
        self._check_input()

    def _check_input(self) -> None:
        """Checks if the input path exists"""
        if not os.path.exists(self.input):
            raise FileNotFoundError('No such file or directory')

    @abstractmethod
    def run(self) -> None:
        """Reads input files and Write the prediction results on them"""

    @abstractmethod
    def _get_yolo_predictions(self,):
        """
        Returns YOLOv8 predictions for an image or video.
        """

    @abstractmethod
    def _visualize_predictions(self,):
        """
        Visualizes YOLO predictions on an image or video.
        """

    @staticmethod
    def _crop_plate(image: np.ndarray, *bbx: list) -> np.ndarray:
        """
        Crops car plate from original image
        """
        plate_crop = image[bbx[0][1]: bbx[0][3], bbx[0][0]: bbx[0][2]]
        return plate_crop

    def _read_plate(self, image: np.ndarray, *bbx: list) -> Tuple:
        """
        Get the results from easyOCR for each image
        """
        plate_crop = self._crop_plate(image, bbx)
        ocr_result = self.reader.readtext(plate_crop,
                                          allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return ocr_result
