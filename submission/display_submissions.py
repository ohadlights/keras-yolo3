import os
import argparse
import cv2
from submission.utils import parse_prediction_line

def main(args):

    content = [l.strip().split(',') for l in open(args.submission_file).readlines()[1:]]
    for l in content:
        path = os.path.join(args.images_dir, l[0] + '.jpg')
        boxes_data = l[1]

        image = cv2.imread(path)
        h, w, _ = image.shape

        boxes = parse_prediction_line(boxes_data, w=w, h=h, to_ints=True)
        for box in boxes:
            image = cv2.rectangle(image, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 255, 0), 2)

        cv2.imshow('Detections', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_file', default=r'merged\2018-08-24_20-10-00.csv')
    parser.add_argument('--images_dir', default=r'D:\Projects\OpenImagesChallenge\oid\challenge2018')
    main(parser.parse_args())