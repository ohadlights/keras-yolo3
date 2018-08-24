import os
import argparse


def create_list_from_classes(class_ids, )
    pass


def main(args):
    class_id_to_index = {}

    train_list_content = open(r'').readlines()
    val_list_content = open(r'').readlines()

    for l in train_list_content:
        l = l.strip().split()
        path = l[0]

        

    # for i in range(args.num_groups):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_groups', default=5)
    parser.add_argument('--classes_path', default=r'..\..\model_data\oid_classes.txt')
    main()