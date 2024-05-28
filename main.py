from omegaconf import OmegaConf

from data_preparation import PrepareData, TrainTestSplit
from pipeline import Pipeline

def main():
    Config = OmegaConf.load('config.yaml')
    images_dir = Config.images_dir
    labels_dir = Config.labels_dir
    raw_annot = Config.raw_annot
    PrepareData(images_dir, labels_dir, raw_annot)
    TrainTestSplit().split()
    Pipeline(config_path='config.yaml').run()


if __name__ == '__main__':
    main()