import os
import argparse


coco_classes = [l.strip() for l in open(r'..\model_data\coco_classes.txt')]


def main(args):

    # build maps

    coco_map = {coco_classes[i].lower(): i for i in range(len(coco_classes))}

    content = [l.strip().lower() for l in open(args.oid_classes_path).readlines()]
    oid_map = {content[i]: i for i in range(len(content))}

    coco_to_oid_class_id = {}
    for name, id in coco_map.items():
        if name in oid_map:
            coco_to_oid_class_id[id] = oid_map[name]

    # find relevant images

    coco_annotations = [l.strip() for l in open(args.coco_annotations_file).readlines()]
    with open(args.output_path, 'w') as f:
        for a in coco_annotations:
            boxes = a.split()[1:]
            filtered_boxes = []
            for box in boxes:
                box = box.split(',')
                class_id = int(box[4])
                if class_id in coco_to_oid_class_id:
                    box[4] = str(coco_to_oid_class_id[class_id])
                    filtered_boxes += [','.join(box)]
            if len(filtered_boxes) > 0:
                f.write('{}{} {}\n'.format(args.relative_path, os.path.basename(a.split()[0]), ' '.join(filtered_boxes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oid_classes_path', default=r'..\model_data\oid_classes.txt')
    parser.add_argument('--output_path', default=r'..\data\image_list_coco_2007_train.txt')
    parser.add_argument('--coco_annotations_file', default=r'..\data\train_coco_2017.txt')
    parser.add_argument('--relative_path', default=r'COCO/2017/train2017/')
    main(parser.parse_args())