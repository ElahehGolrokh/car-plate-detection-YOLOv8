# car-plate-detection-YOLOv8
Welcome to the car plate detection using YOLOv8 🚀! The basic usage is based on the YOLOv8 tutorial and has been customized for the current dataset to guide you step-by-step in preparing data and training an object detection model using YOLO. <br>
This tutorial will take you from installation, to  training YOLOv8 object detection model with a custom dataset, then exporting it for inference.

## Installation 

This package is tested in ubuntu 20.04 with python 3.9.12. First of all creat your virtual environment:

```shell
python -m venv venv
source venv/bin/activate
```

Next for installing all dependencies run this command:

```shell
pip install -r requirements.txt
```

## Train YOLOv8 object detection model on a custom dataset

For training your own object detection model you can run ```shell main.py```. There are some arguments you can customize: <br>
-rpr or --remove_prev_runs: whether you want to remove previous runs <br>
-p or --prepare: whether you want to implement data preparation <br>
-t or --train: whether you want to implement training <br>
-e or --export: whether you want to export a saved model <br>

### Data Preparation

In order to use YOLOv8 for your object detection task, you have to structure your data in a specific way. <br>

1. First of all in the root directory of your dataset, you must have two folders with these specific names: images and labels. In this tutorial we consider data/ in the root of our project as the root of our dataset. 
2. Images could be in any of jpg or png formats.
3. There must be a config file in the yaml format in which the paths to the root and images directories are specified
4. Separating train, validation and test partitions is optional. If you want to do this, you have to consider these subdirectories in both folders, images and labels. If that's the case, all these paths have to be specified in the config file.
The labels have to be in txt format and for each bounding box in the image, there must be a row in the corresponding label file with the following structure without any commos: class_label, bbx_x_center, bbx_y_center, bbx_width, bbx_height <br>

The data directory has to contain components like this: <br>
── data
│   ├── images
│   │   ├── test
│   │   │   ├── Cars27.png
│   │   │   ├── .
│   │   │   ├── .
│   │   ├── train
│   │   │   ├── Cars0.png
│   │   │   ├── .
│   │   │   ├── .
│   │   └── validation
│   │       ├── Cars10.png
│   │       ├── .
│   │       ├── .
│   └── labels
│       ├── test
│       │   ├── Cars27.txt
│       │   ├── .
│       │   ├── .
│       ├── train
│       │   ├── Cars0.txt
│       │   ├── .
│       │   ├── .
│       ├── validation
│       │   ├── Cars10.txt
│       │   ├── .
│       │   ├── .


## Inference with trained models

For getting predictions from a YOLO saved model, you can run this command:

```shell
python inference.py --model_path 'path/to/model' --image_path 'path/to/test_image'
```
The default path for saved model could be: runs/detect/train/weights/best.pt <br>
Besides, the test image could be jpg or png. The result of model predicted bounding boxes will be saved in runs directory as a png file.

