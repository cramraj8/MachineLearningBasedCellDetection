# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import sys
import numpy as np
import cv2
import os
import glob
import params
from sklearn.utils import shuffle
import sys


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(imgs, lbls, IDs, tfrecord_name='./wbc.tfrecords'):

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    n_obs = imgs.shape[0]
    for i in range(n_obs):
        if not i % 1000:
            print('Data: {}/{}'.format(i, n_obs))
            sys.stdout.flush()

        # Load the image
        img = imgs[i, :, :, :]

        encoded_label = lbls[i, :]
        label = np.argmax(encoded_label)
        ID = IDs[i]

        # Create a feature
        # feature = {'{}/label'.format(lbl): _int64_feature(label),
        #            '{}/image'.format(lbl): _bytes_feature(tf.compat.as_bytes(
        #                img.tostring())),
        #            '{}/id'.format(lbl): _bytes_feature(tf.compat.as_bytes(ID))}
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                   'train/id': _bytes_feature(tf.compat.as_bytes(ID))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def load_slidewise_data(data=FLAGS.data_dir):
    images = []
    labels = []
    ids = []
    cls = []

    CLASSES_OBJ = ['objectness']

    print('Reading training images')
    for i, fld in enumerate(CLASSES_OBJ):
        index = CLASSES_OBJ.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(data, fld, '*g')
        files = glob.glob(path)
        files = shuffle(files, random_state=FLAGS.seed)

        for fl in files:
            fl_slide = fl.split('/')[-1].split('.')[1]
            image = cv2.imread(fl)
            # image = cv2.resize(image,
            #                    (FLAGS.orig_size, FLAGS.orig_size),
            #                    interpolation=cv2.INTER_LINEAR)
            # Do you want random crop augmentation or not
            image = cv2.resize(image,
                               (FLAGS.img_size, FLAGS.img_size),
                               interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(CLASSES_OBJ))
            label[index] = 1
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def creat_tf_records():

    # ========================================
    # Load test data
    TFRECORD_DIR = './DISEASED_SAMPLES_TFRECORD_WHOLE_96/'
    DIR = './DISEASED_SAMPLES_CROPPEDCELLS_WHOLE_96/'

    test_images, test_labels, test_ids, test_cls = load_slidewise_data(DIR)
    print('\nTest Data Loaded.')
    print('Test Data : ', test_images.shape, '\n')

    # Shuffle
    test_images, test_labels, test_ids, test_cls = shuffle(test_images,
                                                           test_labels,
                                                           test_ids,
                                                           test_cls)

    if not os.path.exists(TFRECORD_DIR):
        os.makedirs(TFRECORD_DIR)

    print('\n')
    write_record(test_images, test_labels, test_ids,
                 tfrecord_name=TFRECORD_DIR + 'slide_test.tfrecords')


if __name__ == '__main__':
    creat_tf_records()
