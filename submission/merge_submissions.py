import os
import argparse
import datetime
from collections import defaultdict


def bb_intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def main(args):

    # Collect data from submissions

    files = list(filter(lambda f: f.endswith('.csv'), os.listdir('.')))
    image_to_boxes = defaultdict(list)
    for f in files:
        content = open(f).readlines()[1:]
        for l in content:
            info = l.strip().split(',')
            image_id = info[0]
            image_to_boxes[image_id] += [info[1]]

    # output merged submission

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    info_file = os.path.join('merged', current_time + '.txt')
    submission_file = os.path.join('merged', current_time + '.csv')

    with open(info_file, 'w') as f:
        f.write('\n'.join(files))

    with open(submission_file, 'w') as f:
        f.write('ImageId,PredictionString\n')
        for image_id, boxes in image_to_boxes.items():
            f.write('{},'.format(image_id))
            f.write(' '.join(boxes))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
