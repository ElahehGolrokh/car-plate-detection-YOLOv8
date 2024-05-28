import argparse
import os

from prediction import Predict


parser = argparse.ArgumentParser(description='Gets model path and file path')

# Defining the parser arguments
parser.add_argument('-mp',
                    '--model_path',
                    default='/kaggle/working/runs/detect/train/weights/best.pt',
                    help='path to saved YOLOv8 model')
parser.add_argument('-fp',
                    '--image_path',
                    help='path to jpg or png test image')
parser.add_argument('-on',
                    '--output_name',
                    default='output.png',
                    help='name of plt saved figure of final prediction')
args = parser.parse_args()


def main(image_path, model_path, output_name):
    output_path = os.path.join('runs', output_name)
    # load a pretrained model (recommended for training)
    class_names = ['Car Plate']  # Replace with your actual class names
    predict = Predict(image_path, model_path, output_path)
    predictions = predict.get_yolo_predictions()
    # Visualize predictions
    predict.visualize_predictions(predictions, class_names)


if __name__ == '__main__':
    main(args.image_path, args.model_path, args.output_name)
