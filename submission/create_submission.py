"""
For each image in the test set, you must predict a list of boxes describing objects in the image.
Each box is described as <label confidence x_min y_min x_max y_max>.
The length of your PredictionString should always be multiple of 6.
If there are no boxes predicted for a given image, PredictionString should be empty.
Every value is space delimited. The file should contain a header and have the following format:

ImageId,PredictionString
fd162df2a4fdb29d,/m/05s2s 0.9 0.46 0.08 0.93 0.5 /m/0c9ph5 0.5 0.25 0.96 0.6 0.9
...
"""


import os
import argparse

from PIL import Image
from tqdm import tqdm

from submission.yolo import Yolo


def main(args):

    class_descs = {l[1]: l[0] for l in [l.strip().split(',') for l in open(args.class_descriptions_path).readlines()]}

    with Yolo(args.classes_path, args.anchors_path, args.model_path) as yolo:

        model_name = os.path.basename(args.model_path)
        output_path = model_name.replace('.h5', '.csv')

        with open(output_path, 'w') as f:
            f.write('ImageId,PredictionString\n')

            for file in tqdm(os.listdir(args.images_dir)):
                path = os.path.join(args.images_dir, file)
                image = Image.open(path)
                width, height = image.size
                detections = yolo.detect_image(image)

                f.write('{},'.format(file.replace('.jpg', '')))
                for d in detections:
                    d = [
                        class_descs[d[0]],
                        d[1],
                        d[2] / width,
                        d[3] / height,
                        d[4] / width,
                        d[5] / height
                    ]
                    f.write('{} '.format(' '.join([str(a) for a in d])))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=r'X:\OpenImages\yolov3\train\trainep010-loss51.981-val_loss50.800.h5')
    parser.add_argument('--anchors_path', default='..\model_data\yolo_anchors.txt')
    parser.add_argument('--classes_path', default='..\model_data\oid_classes.txt')
    parser.add_argument('--images_dir', default=r'X:\OpenImages\images\challenge2018')
    parser.add_argument('--class_descriptions_path', default=r'X:\OpenImages\docs\challenge-2018-class-descriptions-500.csv')
    main(parser.parse_args())