from omegaconf import OmegaConf

from data_preparation import PrepareData, TrainTestSplit


def main():
    Config = OmegaConf.load('config.yaml')
    images_dir = Config.images_dir
    labels_dir = Config.labels_dir
    raw_annot = Config.raw_annot
    PrepareData(images_dir, labels_dir, raw_annot)
    TrainTestSplit().split()


if __name__ == '__main__':
    main()