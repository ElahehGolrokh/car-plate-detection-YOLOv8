import subprocess

from omegaconf import OmegaConf

# bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

class Pipeline:
    def __init__(self,
                 config_path: str,
                 train: bool = True,
                 export: bool = False,
                 model_path: str = 'runs/detect/train/weights/best.pt',
                 export_format: str = 'torchscript') -> None:
        self.config_path = config_path
        self.train = train
        self.export = export
        self.model_path = model_path
        self.export_format = export_format

    def run(self):
        if self.train:
            Config = OmegaConf.load(self.config_path)
            image_size = Config.image_size
            epochs = Config.epochs
            bashCommand = f"yolo train model=yolov8n.pt data={self.config_path} epochs={epochs} imgsz={image_size}"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        
        if self.export:
            bashCommand = f"yolo export model={self.model_path} format={self.export_format}"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
