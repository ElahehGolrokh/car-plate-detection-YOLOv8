import argparse
import easyocr
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
parser.add_argument('-rf',
                    '--read_flag',
                    action='store_true',  # Default value is False
                    help='specifies whether to read car plates using OCR')
args = parser.parse_args()


def main(image_path, video_path, model_path, output_name, image_dir, read_flag):
    if read_flag:
        reader = easyocr.Reader(['en'])
    else:
        reader = None
    if image_path:
        image_predictor = ImagePredictor(image_path, model_path, output_name, reader)
        image_predictor.run()
    #     output_path = os.path.join('runs', output_name)
    #     class_names = ['Car Plate']  # Replace with your actual class names
    #     image_predictor = ImagePredictor(image_path, model_path, output_path, reader)
    #     predictions = image_predictor.get_yolo_predictions()
        
    #     # Visualize the predictions
    #     image_predictor.visualize_predictions(predictions, class_names)
    
    # elif image_dir:
    #     for file in os.listdir(image_dir):
    #         image_path = os.path.join(image_dir, file)
    #         output_path = os.path.join('runs', file)
    #         class_names = ['Car Plate']  # Replace with your actual class names
    #         image_predictor = ImagePredictor(image_path, model_path, output_path, reader)
    #         predictions = image_predictor.get_yolo_predictions()

    #         # Visualize the predictions
    #         image_predictor.visualize_predictions(predictions, class_names)

    elif video_path:
        output_path = os.path.join('runs', output_name)
        video_predictor = VideoPredictor(video_path,
                                         model_path,
                                         output_path,
                                         reader)
        video_predictor.run()


if __name__ == '__main__':
    main(args.image_path,
         args.video_path,
         args.model_path,
         args.output_name,
         args.image_dir,
         args.read_flag)
