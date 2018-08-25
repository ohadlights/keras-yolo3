import argparse


def main(args):

    desc_content = [l.strip().split(',')[0] for l in open(args.class_desc_path).readlines()]

    train_list_content = open(r'X:\OpenImages\yolov3\image_list_448.txt').readlines()
    val_list_content = open(r'X:\OpenImages\yolov3\image_list_val_448.txt').readlines()

    for group_index in range(0, args.num_groups):
        group_file_path = 'group_{}_of_{}'.format(group_index+1, args.num_groups)
        class_ids = [l.strip() for l in open(group_file_path).readlines()]
        class_indexes = set([desc_content.index(class_id) for class_id in class_ids])
        print('group contains {} classes'.format(len(class_indexes)))

        with open(r'X:\OpenImages\yolov3\lists\groups\group_{}_of_{}_train.txt'.format(group_index+1, args.num_groups), 'w') as f:

            for l in train_list_content:
                line_split = l.strip().split()
                boxes = []
                for b in line_split[1:]:
                    class_index = int(b.split(',')[4])
                    if class_index in class_indexes:
                        boxes += [b]
                if len(boxes) > 0:
                    f.write('{} {}\n'.format(line_split[0], ' '.join(boxes)))

        with open(r'X:\OpenImages\yolov3\lists\groups\group_{}_of_{}_val.txt'.format(group_index+1, args.num_groups), 'w') as f:

            for l in val_list_content:
                line_split = l.strip().split()
                boxes = []
                for b in line_split[1:]:
                    class_index = int(b.split(',')[4])
                    if class_index in class_indexes:
                        boxes += [b]
                if len(boxes) > 0:
                    f.write('{} {}\n'.format(line_split[0], ' '.join(boxes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_groups', default=5)
    parser.add_argument('--class_desc_path', default=r'X:\OpenImages\docs\challenge-2018-class-descriptions-500.csv')
    main(parser.parse_args())