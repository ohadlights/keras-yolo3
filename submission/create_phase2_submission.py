"""
For each image in the test set, you must predict a list of boxes describing objects in the image.
Each box is described as:
    <confidence label_object1 x_min1 y_min1 x_max1 y_max1 label_object2 x_min2 y_min2 x_max2 y_max2 relationship_label>.
The length of your PredictionString should always be a multiple of 12.
Every value is space delimited. The file should contain a header and have the following format:

ImageId,PredictionString
fd162df2a4fdb29d,0.037432 /m/03bt1vf 0.549840 0.603769 0.814588 0.999519 /m/01mzp 0.187824 0.454496 0.245905 0.554354 on 0.044382 /m/03bt1vf 0.549840 0.603769 0.814588 0.999519 /m/01mzpv 0.174735 0.468313 0.238807 0.562794 on
...
"""


import os
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
