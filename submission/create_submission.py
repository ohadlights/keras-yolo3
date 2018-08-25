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
from multiprocessing import Pool
from functools import partial

from PIL import Image
from tqdm import tqdm
import numpy as np

from submission.yolo import Yolo
from yolo3.utils import letterbox_image


def preprocess_image(image, model_image_size, adjust_dims=True):
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    # print(image_data.shape)
    image_data /= 255.
    if adjust_dims:
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image_data


def load_image(path, model_image_size):
    image = Image.open(path)
    width, height = image.size
    image_data = preprocess_image(image, model_image_size)
    return image, image_data, width, height


def main(args):

    class_descs = {l[1]: l[0] for l in [l.strip().split(',') for l in open(args.class_descriptions_path).readlines()]}

    model_name = args.model_path.split('\\')[-2]
    checkpoint_name = os.path.basename(args.model_path)
    output_path = os.path.join('submission_files', model_name + '_' + checkpoint_name.replace('.h5', '.csv'))

    with Yolo(args.classes_path, args.anchors_path, args.model_path) as yolo:

        load_image_func = partial(load_image, model_image_size=yolo.model_image_size)

        with open(output_path, 'w') as f, Pool(args.num_processes) as p:
            f.write('ImageId,PredictionString\n')

            files = os.listdir(args.images_dir)
            chunk_size = args.preprocess_chunk_size

            for chunk in tqdm(range(0, len(files), chunk_size)):
                chunk_flies = files[chunk:chunk+chunk_size]
                chunk_paths = [os.path.join(args.images_dir, file) for file in chunk_flies]
                chunk_images = p.map(load_image_func, chunk_paths)

                for i in range(len(chunk_images)):
                    file = chunk_flies[i]
                    image, image_data, width, height = chunk_images[i]
                    detections = yolo.detect_image(image=image, image_data=image_data)

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
    parser.add_argument('--model_path', default=r"Y:\OpenImages\yolov3\models\train_group_5_of_5\ep001-loss33.294-val_loss32.247.h5")
    parser.add_argument('--anchors_path', default='..\model_data\yolo_anchors.txt')
    parser.add_argument('--classes_path', default='..\model_data\oid_classes.txt')
    parser.add_argument('--images_dir', default=r'D:\FaceWS\keras-yolo3\challenge2018')
    parser.add_argument('--class_descriptions_path', default=r'Y:\OpenImages\docs\challenge-2018-class-descriptions-500.csv')
    parser.add_argument('--preprocess_chunk_size', type=int, default=100)
    parser.add_argument('--num_processes', type=int, default=25)
    main(parser.parse_args())
