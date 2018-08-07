import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import cv2


'''
annotations file format:
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
8d6dec80235b6fea,xclick,/m/09j5n,1,0.76,0.778125,0.645892,0.673277,0,0,0,0,0
8d6dec80235b6fea,xclick,/m/09j5n,1,0.8175,0.831875,0.628895,0.661945,0,0,0,0,0

output file format:
One row for one image;
Row format: image_file_path box1 box2 ... boxN;
Box format: x_min,y_min,x_max,y_max,class_id (no space)
Here is an example:
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
'''


def main(args):
    classes = [l.strip().split(',')[0] for l in open(args.class_description_path).readlines()]

    annotations = defaultdict(list)

    content = [l.strip().split(',') for l in open(args.oid_annotations_csv_path).readlines()[1:]]

    for l in tqdm(content):
        image_id = l[0]
        class_id = classes.index(l[2])
        box = [float(a) for a in l[4:8]]
        x_min = box[0]
        x_max = box[1]
        y_min = box[2]
        y_max = box[3]
        annotations[image_id] += [(x_min, y_min, x_max, y_max, class_id)]

    with open(args.output_list_path, 'w') as f:
        for image_id, boxes in tqdm(annotations.items()):
            image_path = os.path.join(os.path.join(args.images_dir, image_id) + '.jpg')
            if os.path.exists(image_path):
                h, w, _ = cv2.imread(image_path).shape
                if args.linux_images_dir:
                    image_path = os.path.join(os.path.join(args.linux_images_dir, image_id) + '.jpg')
                f.write(image_path)
                for box in boxes:
                    f.write(' {},{},{},{},{}'.format(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h), box[4]))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_description_path', default=r'X:\OpenImages\docs\challenge-2018-class-descriptions-500.csv')
    parser.add_argument('--oid_annotations_csv_path', default=r'X:\OpenImages\docs\challenge-2018-train-annotations-bbox---no_val.csv')
    parser.add_argument('--images_dir', default=r'X:\OpenImages\images\train')
    parser.add_argument('--linux_images_dir')  # , default='/mnt/face-public/OpenImages/images/train')
    parser.add_argument('--output_list_path', default=r'D:\Projects\OpenImagesChallenge\keras-yolo3\data\image_list.txt')
    main(parser.parse_args())
