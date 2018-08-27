import random


map_oid_id_to_coco_id = {l[1]: l[0] for l in [l.strip().split(',') for l in open(r'..\model_data\coco_to_oid.txt').readlines()]}

content = open(r"X:\OpenImages\yolov3\lists\image_list_448_balanced_v2.txt").readlines()
with open(r'X:\OpenImages\yolov3\lists\for_coco\image_list_448_balanced_v2_coco.txt', 'w') as f:
    for l in content:
        filtered = []
        boxes = l.strip().split()[1:]
        for b in boxes:
            b = b.split(',')
            class_id = b[4]
            if class_id in map_oid_id_to_coco_id:
                b[4] = map_oid_id_to_coco_id[class_id]
                filtered += [','.join(b)]
        if len(filtered) > 0:
            f.write('{}'.format(l.strip().split()[0]))
            for b in filtered:
                f.write(' {}'.format(b))
            f.write('\n')
        elif random.uniform(0, 1) > 0.8:
            f.write('{}\n'.format(l.strip().split()[0]))
