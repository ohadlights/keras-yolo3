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


def get_files_list(args):
    if args.images_list:
        files = [l.strip().split()[0] for l in open(args.images_list).readlines()]
    else:
        files = os.listdir(args.images_dir)
    return [os.path.join(args.images_dir, file) for file in files]


def main(args):

    class_descs = {l[1]: l[0] for l in [l.strip().split(',') for l in open(args.class_descriptions_path).readlines()]}

    id_map = {int(l[0]): int(l[1]) for l in [l.strip().split(',') for l in open(args.id_map_path).readlines()]}

    coco_classes = [l.strip() for l in open(args.classes_path)]
    coco_name_to_id = {coco_classes[i].lower(): i for i in range(len(coco_classes))}

    oid_clases = [l.strip() for l in open(args.oid_classes_path)]

    with Yolo(args.classes_path, args.anchors_path, args.model_path) as yolo:

        model_name = args.model_path.split('/')[-2] if args.linux else args.model_path.split('\\')[-2]
        checkpoint_name = os.path.basename(args.model_path)
        output_path = os.path.join('submission_files', model_name + '_' + checkpoint_name.replace('.h5', '.csv'))

        with open(output_path, 'w') as f:
            f.write('ImageId,PredictionString\n')

            for path in tqdm(get_files_list(args)):
                image = Image.open(path)
                width, height = image.size
                detections = yolo.detect_image(image)

                f.write('{},'.format(os.path.basename(path).replace('.jpg', '')))
                for d in detections:
                    coco_class_id = coco_name_to_id[d[0]]
                    oid_id = id_map[coco_class_id]
                    if oid_id < 999:
                        oid_class_name = oid_clases[oid_id]
                        class_desc = class_descs[oid_class_name] if oid_class_name in class_descs else oid_class_name
                        # print('{} -> {}'.format(d[0], oid_class_name))
                        d = [
                            class_desc,
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
    parser.add_argument('--model_path', default=r"X:\OpenImages\yolov3\models\no_coco_3_of_3\ep003-loss28.185-val_loss28.048.h5")
    parser.add_argument('--anchors_path', default='..\model_data\yolo_anchors.txt')
    parser.add_argument('--classes_path', default='..\model_data\coco_classes.txt')
    parser.add_argument('--images_dir', default=r'D:\Projects\OpenImagesChallenge\oid\challenge2018')
    parser.add_argument('--images_list')
    parser.add_argument('--class_descriptions_path', default=r'X:\OpenImages\docs\challenge-2018-class-descriptions-500.csv')
    parser.add_argument('--linux', action='store_true')

    parser.add_argument('--id_map_path', default=r'..\model_data\coco_to_oid.txt')
    parser.add_argument('--oid_classes_path', default=r'..\model_data\oid_classes.txt')

    main(parser.parse_args())
