import os
import argparse
import datetime
from collections import defaultdict


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
