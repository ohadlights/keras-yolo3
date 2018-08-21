import argparse
import random
from collections import defaultdict
from tqdm import tqdm


def main(args):

    # Read training list

    content = [l.strip().split() for l in open(args.input_list).readlines()]

    # Extract images per class from training list

    images_per_class = defaultdict(list)
    for l in tqdm(content):
        boxes = l[1:]
        classes_in_image = set()
        for box in boxes:
            class_id = box.split(',')[4]
            classes_in_image.add(class_id)
        for c in classes_in_image:
            images_per_class[c] += [(l[0], l)]

    # Compress list

    written_images = set()
    with open(args.output_list, 'w') as f:
        for c, images in tqdm(images_per_class.items()):
            random.shuffle(images)
            images = images[:args.take_per_class]
            for image, line in images:
                if image not in written_images:
                    f.write('{}\n'.format(' '.join(line)))
                    written_images.add(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_list', default=r'X:\OpenImages\yolov3\image_list_train_full_v2.txt')
    parser.add_argument('--output_list', default=r'data\small_image_list.txt')
    parser.add_argument('--take_per_class', type=int, default=300)
    main(parser.parse_args())