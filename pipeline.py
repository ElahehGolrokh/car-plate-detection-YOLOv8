import os
import shutil
import subprocess

from omegaconf import OmegaConf

from data_preparation import PrepareData, TrainTestSplit


class Pipeline:
    def __init__(self,
                 config_path: str,
                 remove_prev_runs: bool = True,
                 prepare: bool = True,
                 train: bool = True,
                 export: bool = False,
                 model_path: str = 'runs/detect/train/weights/best.pt',
                 export_format: str = 'torchscript') -> None:
        self.config_path = config_path
        self.remove_prev_runs = remove_prev_runs
        self.prepare = prepare
        self.train = train
        self.export = export
        self.model_path = model_path
        self.export_format = export_format

    def run(self):
        Config = OmegaConf.load(self.config_path)
        if self.remove_prev_runs:
            shutil.rmtree('runs', )
            # os.removedirs('runs')
        if self.prepare:
            images_dir = Config.images_dir
            labels_dir = Config.labels_dir
            raw_annot = Config.raw_annot
            PrepareData(images_dir, labels_dir, raw_annot)
            TrainTestSplit().split()
        if self.train:
            image_size = Config.image_size
            epochs = Config.epochs
            bashCommand = f"yolo train model=yolov8n.pt data={self.config_path} epochs={epochs} imgsz={image_size}"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        
        if self.export:
            bashCommand = f"yolo export model={self.model_path} format={self.export_format}"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
