import argparse
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
parser.add_argument('-vp',
                    '--video_path',
                    default=None,
                    help='path to mp4 test video')
parser.add_argument('-on',
                    '--output_name',
                    default='output.png',
                    help='name of plt saved figure or viseo of the final prediction')
args = parser.parse_args()


def main(image_path, video_path, model_path, output_name):
    output_path = os.path.join('runs', output_name)
    if image_path:
        class_names = ['Car Plate']  # Replace with your actual class names
        image_predictor = ImagePredictor(image_path, model_path, output_path)
        predictions = image_predictor.get_yolo_predictions()
        # Visualize the predictions
        image_predictor.visualize_predictions(predictions, class_names)
    if video_path:
        video_predictor = VideoPredictor(video_path=video_path,
                                         model_path=model_path,
                                         output_path=output_path)
        video_predictor.run()


if __name__ == '__main__':
    main(args.image_path, args.video_path, args.model_path, args.output_name)
