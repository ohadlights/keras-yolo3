from tqdm import tqdm


lists = [r'X:\OpenImages\yolov3\image_list_relative.txt',
         r'X:\OpenImages\yolov3\image_list_voc_2007_train.txt',
         r'X:\OpenImages\yolov3\image_list_voc_2007_val.txt']

print('merging {} lists'.format(len(lists)))

with open(r'X:\OpenImages\yolov3\image_list_train_full.txt', 'w') as f:
    for l in lists:
        content = open(l).readlines()
        for l in tqdm(content):
            f.write(l)
