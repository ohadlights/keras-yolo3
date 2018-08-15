import os
import argparse


voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def main(args):

    # build maps

    voc_map = {voc_classes[i].lower(): i for i in range(len(voc_classes))}

    content = [l.strip().lower() for l in open(args.oid_classes_path).readlines()]
    oid_map = {content[i]: i for i in range(len(content))}

    voc_to_oid_class_id = {}
    for name, id in voc_map.items():
        if name in oid_map:
            voc_to_oid_class_id[id] = oid_map[name]

    # find relevant images

    voc_annotations = [l.strip() for l in open(args.voc_annotations_file).readlines()]
    with open(args.output_path, 'w') as f:
        for a in voc_annotations:
            boxes = a.split()[1:]
            filtered_boxes = []
            for box in boxes:
                box = box.split(',')
                class_id = int(box[4])
                if class_id in voc_to_oid_class_id:
                    box[4] = str(voc_to_oid_class_id[class_id])
                    filtered_boxes += [','.join(box)]
            if len(filtered_boxes) > 0:
                f.write('{}{} {}\n'.format(args.relative_path, os.path.basename(a.split()[0]), ' '.join(filtered_boxes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oid_classes_path', default=r'..\model_data\oid_classes.txt')
    parser.add_argument('--output_path', default=r'..\data\image_list_voc_2007_train.txt')
    parser.add_argument('--voc_annotations_file', default=r'..\data\2007_train.txt')
    parser.add_argument('--relative_path', default=r'VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/')
    main(parser.parse_args())