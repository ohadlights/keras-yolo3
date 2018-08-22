import os
from PIL import Image
import numpy as np
import cv2
from yolo import Yolo, detect_video
from yolo_args import build_args


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()


def detect_from_files(yolo, files):
    for path in files:
        image = Image.open(path)
        r_image = yolo.detect_image(image)
        ImageNumpyFormat = np.asarray(r_image)[...,::-1]
        cv2.imshow('Viewer', ImageNumpyFormat)
        cv2.waitKey(0)


def detect_from_directory(yolo, directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    detect_from_files(yolo, files)


def detect_from_list(yolo, list_path, images_root_dir):
    prefix = images_root_dir if images_root_dir else ''
    files = [os.path.join(prefix, l.strip().split()[0]) for l in open(list_path).readlines()]
    detect_from_files(yolo, files)


def get_images_source(args):
    if args.input == 'image':
        return 'image'
    if args.input.endswith('txt'):
        return 'list'
    else:
        return 'directory'


if __name__ == '__main__':
    parser = build_args()

    parser.add_argument('--images_root_dir')

    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=True, default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    args = parser.parse_args()

    images_source = get_images_source(args)

    with Yolo(args) as yolo:

        if images_source == 'directory':
            print("Images from directory mode")
            detect_from_directory(yolo, args.input)
        elif images_source == 'list':
            print("Images from list mode")
            detect_from_list(yolo, args.input, args.images_root_dir)
        elif images_source == 'image':
            """
            Image detection mode, disregard any remaining command line arguments
            """
            print("Image detection mode")
            if "input" in args:
                print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
            detect_img(yolo)
        # elif "input" in args:
        #     detect_video(yolo, args.input, args.output)
        else:
            print("Must specify at least video_input_path.  See usage with --help.")
