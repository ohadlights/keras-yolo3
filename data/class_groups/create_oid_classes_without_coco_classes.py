map_oid_id_to_coco_id = {int(l[1]): l[0] for l in [l.strip().split(',') for l in open(r'..\..\model_data\coco_to_oid.txt').readlines()]}

shared_classes_ids = list(filter(lambda a: a < 999, map_oid_id_to_coco_id.keys()))

class_descs = open(r'X:\OpenImages\docs\challenge-2018-class-descriptions-500.csv').readlines()

with open('not_in_coco.txt', 'w') as f:
    for i in range(len(class_descs)):
        if i not in shared_classes_ids:
            f.write(class_descs[i])
