import argparse


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required=True)
    parser.add_argument('--anchors', default='model_data/yolo_anchors.txt')
    parser.add_argument('--classes', default=r'D:\Projects\OpenImagesChallenge\keras-yolo3\model_data\oid_classes.txt')

    parser.add_argument('--score', default=0.3)
    parser.add_argument('--iou', default=0.45)

    parser.add_argument('--gpu_num', default=1)

    return parser