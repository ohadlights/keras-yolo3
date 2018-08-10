import threading
import numpy as np
from yolo3.utils import get_random_data
from yolo3.model import preprocess_true_boxes
from keras.utils import Sequence


'''
Use generator!
'''

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


# @threadsafe_generator
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


'''
Use Sequence!
'''


class YoloSequence(Sequence):

    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        # self.x, self.y = x_set, y_set
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes

    def __len__(self):
        return int(np.floor(len(self.annotation_lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx == 0:
            np.random.shuffle(self.annotation_lines)

        image_data = []
        box_data = []

        start_index = idx * self.batch_size
        end_index = (idx + 1) * self.batch_size

        for index in range(start_index, end_index):
            image, box = get_random_data(self.annotation_lines[index], self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)

        return [image_data, *y_true], np.zeros(self.batch_size)


'''
wrapper
'''


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    # return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
    return YoloSequence(annotation_lines, batch_size, input_shape, anchors, num_classes)
