import os
import argparse
import cv2


def main(args):

    content = [l.strip().split(',') for l in open(args.submission_file).readlines()[1:]]
    for l in content:
        path = os.path.join(args.images_dir, l[0] + '.jpg')
        boxes_data = l[1].split()

        image = cv2.imread(path)
        h, w, _ = image.shape

        for i in range(0, len(boxes_data), 6):
            box = [
                int(float(boxes_data[i+2]) * w),
                int(float(boxes_data[i+3]) * h),
                int(float(boxes_data[i+4]) * w),
                int(float(boxes_data[i+5]) * h),
            ]
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0 ,255, 0), 2)

        cv2.imshow('Detections', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_file', default=r'merged\2018-08-24_12-32-10.csv')
    parser.add_argument('--images_dir',
                        default=r'\\ger\ec\proj\ha\RSG\FacePublicDatasets\OpenImages\images\challenge2018')
    main(parser.parse_args())