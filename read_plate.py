import argparse
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os

from src.prediction import ImagePredictor, VideoPredictor


parser = argparse.ArgumentParser(description='Gets model path, file path and' +
                                             'a name for saving output.')

# Defining the parser arguments
parser.add_argument('-mp',
                    '--model_path',
                    default='runs/detect/train/weights/best.pt',
                    help='path to saved YOLOv8 model')
parser.add_argument('-ip',
                    '--image_path',
                    default=None,
                    help='path to jpg or png test image')
parser.add_argument('-imgd',
                    '--image_dir',
                    default=None,
                    help='path to test images directory')
parser.add_argument('-vp',
                    '--video_path',
                    default=None,
                    help='path to mp4 test video')
parser.add_argument('-on',
                    '--output_name',
                    default='output.png',
                    help='name of plt saved figure or viseo of the final prediction')
args = parser.parse_args()


def get_plates_xy(image_path: np.ndarray, bbx: list, reader: easyocr.Reader) -> tuple:
    '''Get the results from easyOCR for each frame and return them with bounding box coordinates'''
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x_min, y_min, x_max, y_max = (int(bbx[0]),
                                  int(bbx[1]),
                                  int(bbx[2]),
                                  int(bbx[3])) ## BBOx coordniates

    plate_crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    print('plate_crop shape = ', plate_crop.shape)
    plt.imshow(plate_crop)
    plt.show()
    ocr_result = reader.readtext(plate_crop,
                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')  #, paragraph="True", min_size=50)
    
    return ocr_result, x_min, y_min


def main(image_path, video_path, model_path, output_name, image_dir):
    if image_path:
        output_path = os.path.join('runs', output_name)
        class_names = ['Car Plate']  # Replace with your actual class names
        image_predictor = ImagePredictor(image_path, model_path, output_path)
        predictions = image_predictor.get_yolo_predictions()

        reader = easyocr.Reader(['en'])

        for prediction in predictions:
            ocr_result, _, _ = get_plates_xy(image_path, prediction, reader)
            print(f'OCR RESULT = {ocr_result}')
        
        # Visualize the predictions
        image_predictor.visualize_predictions(predictions, class_names, ocr_result)
    
    elif image_dir:
        for file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, file)
            output_path = os.path.join('runs', file)
            class_names = ['Car Plate']  # Replace with your actual class names
            image_predictor = ImagePredictor(image_path, model_path, output_path)
            predictions = image_predictor.get_yolo_predictions()

            reader = easyocr.Reader(['en'])

            for prediction in predictions:
                ocr_result, _, _ = get_plates_xy(image_path, prediction, reader)
                print(f'OCR RESULT = {ocr_result}')

            # Visualize the predictions
            image_predictor.visualize_predictions(predictions, class_names, ocr_result)
    elif video_path:
        output_path = os.path.join('runs', output_name)
        video_predictor = VideoPredictor(video_path=video_path,
                                         model_path=model_path,
                                         output_path=output_path)
        video_predictor.run()


if __name__ == '__main__':
    main(args.image_path,
         args.video_path,
         args.model_path,
         args.output_name,
         args.image_dir)
