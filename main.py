from data_preparation import PrepareData, TrainTestSplit


def main():
    images_dir = 'data/images'
    labels_dir = 'data/labels'
    raw_annot = 'data/annotations'
    PrepareData(images_dir, labels_dir, raw_annot)
    TrainTestSplit().split()


if __name__ == '__main__':
    main()