import cv2
import os

from bs4 import BeautifulSoup


class PrepareData:
    def __init__(self, images_dir: str, labels_dir: str, raw_annot: str) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.raw_annot = raw_annot
        self._normalize_dataset()
    
    def prepare_pipeline(self):
        pass
    
    def _create_dirs(self) -> None:
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def _modify_xmls(self):
        """
        Converts xml contents to a plain text containing only bbx class and locations
        """
        self._create_dirs()
        if len(os.listdir(self.labels_dir)) == 0:
            print('Convert xml contents to a plain text containing only bbx class and locations. \n',
                'It may take a while ...')
            for file in os.listdir(self.raw_annot):
                filepath = os.path.join(self.raw_annot, file)
                bbxs = []
                with open(filepath, 'r') as f:
                    content = f.read()
                    bs_data = BeautifulSoup(content, "xml")
                    xmins = bs_data.find_all('xmin')
                    xmaxs = bs_data.find_all('xmax')
                    ymins = bs_data.find_all('ymin')
                    ymaxs = bs_data.find_all('ymax')
                    f.close()
                for i in range(len(xmins)):
                    xmin = int(xmins[i].text)
                    xmax = int(xmaxs[i].text)
                    ymin = int(ymins[i].text)
                    ymax = int(ymaxs[i].text)
                    object_class = 0
                    bbx_x_center = (xmax + xmin)/2
                    bbx_y_center = (ymax + ymin)/2
                    bbx_width = xmax - xmin
                    bbx_height = ymax - ymin
                    bbxs.append([bbx_x_center, bbx_y_center, bbx_width, bbx_height])

                filepath = os.path.join(self.labels_dir, file)
                # !touch {filepath}
                # open(filepath, filepath).close()
                with open(filepath, 'w') as f:
                    # for label in labels:
                    for bbx in bbxs:
                        f.write('{} '.format(object_class))
                        f.write('{} '.format(bbx[0]))
                        f.write('{} '.format(bbx[1]))
                        f.write('{} '.format(bbx[2]))
                        f.write('{}'.format(bbx[3]))
                        f.write('\n')
                        f.close()
            print('Done!')
        else:
            print('The labels are already in plain texts as .xml files.')
    
    def _xml_to_txt(self) -> None:
        """Convert labels from xml to txt"""
        self._modify_xmls()
        labels = os.listdir(self.labels_dir)
        labels_check = [True if l.endswith('xml') else False for l in labels]
        if all(labels_check):
            for filename in labels:
                filepath = os.path.join(self.labels_dir, filename)
                if not os.path.isdir(filepath) and filepath.endswith('.xml'):
                    with open(filepath, 'r') as f:
                        content = f.read()
                        f.close()
                    txt_file = filepath.replace('xml', 'txt')
                    # !touch {txt_file}
                    with open(txt_file, 'w') as f:
                        f.write(content)
                    os.remove(filepath)
        else:
            print('The labels are already in txt formats.')

        # Test
        for i, filename in enumerate(os.listdir(self.labels_dir)):
            if filename.endswith('.xml'):
                raise ValueError('There are xml files in labels folder.')

        print(f'Number of label files with .txt format: {len(os.listdir(self.labels_dir))}')
    
    @staticmethod
    def _normalize_coordinates(image_path, bbox):
        """
        Normalize bounding box coordinates.

        :param image_path: Path to the image file.
        :param bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
        :return: Normalized bounding box coordinates [x_min_norm, y_min_norm, x_max_norm, y_max_norm].
        """
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        x_min, y_min, x_max, y_max = bbox
        x_min_norm = x_min / width
        y_min_norm = y_min / height
        x_max_norm = x_max / width
        y_max_norm = y_max / height

        # Ensure coordinates are within [0, 1]
        x_min_norm = min(max(x_min_norm, 0), 1)
        y_min_norm = min(max(y_min_norm, 0), 1)
        x_max_norm = min(max(x_max_norm, 0), 1)
        y_max_norm = min(max(y_max_norm, 0), 1)

        return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

    def _normalize_dataset(self, output_dir=None):
        """
        Normalize bounding box coordinates for the entire dataset.

        :param output_dir: Directory to save the normalized labels.
        """
        self._xml_to_txt()
        if not output_dir:
            output_dir = self.labels_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for label_file in os.listdir(self.labels_dir):
            label_path = os.path.join(self.labels_dir, label_file)
            image_path = os.path.join(self.images_dir,
                                      label_file.replace('.txt', '.png'))

            if not os.path.exists(image_path):
                continue

            with open(label_path, 'r') as file:
                lines = file.readlines()

            normalized_lines = []
            normalized = True
            for line in lines:
                class_id, x_min, y_min, x_max, y_max = map(float,
                                                           line.strip().split())
                norm_check = [True for item in [x_min, y_min, x_max, y_max] if item > 1]
                # If any of items are greater than 1 it means that they are not normalized yet
                if any(norm_check):
                    normalized_bbox = self._normalize_coordinates(image_path,
                                                                [x_min, y_min, x_max, y_max])
                    normalized_lines.append(f"{class_id} {' '.join(map(str, normalized_bbox))}\n")
                else:
                    normalized = False
                    break
            if normalized:
                output_path = os.path.join(output_dir, label_file)
                with open(output_path, 'w') as file:
                    file.writelines(normalized_lines)
            else:
                print('The labels are already normalized.')
                break


class TrainTestSplit:
    def __init__(self) -> None:
        pass