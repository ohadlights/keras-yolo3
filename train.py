"""
Retrain the YOLO model for your own dataset.
"""

import argparse
import numpy as np

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.training_utils import multi_gpu_model

from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from data_generator import data_generator_wrapper


def _main(args):
    annotation_path_train = args.image_list_train
    annotation_path_val = args.image_list_val
    log_dir = args.logs_dir
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=args.weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=args.weights_path) # make sure you know what you freeze
    if args.num_gpus > 1:
        model = multi_gpu_model(model, gpus=args.num_gpus)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    prefix = args.images_dir + '/' if args.images_dir else ''
    with open(annotation_path_train) as f:
        train_lines = [prefix + l for l in f.readlines()]
    with open(annotation_path_val) as f:
        val_lines = [prefix + l for l in f.readlines()]
    num_val = int(len(val_lines))
    num_train = len(train_lines)

    batch_size = args.batch_size
    steps_in_epoch = args.steps_in_epoch if args.steps_in_epoch else max(1, num_train//batch_size)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if args.freeze_backbone:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=steps_in_epoch,
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size) // 2,
                            epochs=50,
                            initial_epoch=args.initial_epoch,
                            workers=args.workers,
                            use_multiprocessing=args.use_multiprocessing,
                            max_queue_size=batch_size * 2,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=steps_in_epoch,
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size) // 2,
                            epochs=100,
                            initial_epoch=args.initial_epoch,
                            workers=args.workers,
                            use_multiprocessing=args.use_multiprocessing,
                            max_queue_size=batch_size * 2,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list_train', required=True)
    parser.add_argument('--image_list_val', required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--anchors_path', type=str, required=True)
    parser.add_argument('--classes_path', default='model_data/oid_classes.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--use_multiprocessing', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--steps_in_epoch', type=int)
    _main(parser.parse_args())
