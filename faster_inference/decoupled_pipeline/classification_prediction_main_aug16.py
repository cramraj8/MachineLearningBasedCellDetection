# -*- coding: utf-8 -*-
# @__ramraj__

from __future__ import division, print_function, absolute_import
from datetime import datetime
import os.path
import re
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import params
import inputs
import pandas as pd
import tensorflow.contrib.slim as slim
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
from statistics import mode
import sys
from glob import glob
import test_inputs


FLAGS = tf.app.flags.FLAGS


def vgg16(inputs,
          batch_size=100,
          num_classes=12,
          is_training=True,
          dropout=0.5,
          weight_decay=0.005,
          spatial_squeeze=True,
          scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    endpoints = {}
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        # end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
                            # ,outputs_collections=end_points_collection):

            net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
            endpoints['conv1_1'] = net
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            endpoints['conv1_2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            endpoints['pool1'] = net

            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            endpoints['conv2_1'] = net
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            endpoints['conv2_2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            endpoints['pool2'] = net

            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            endpoints['conv3_1'] = net
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            endpoints['conv3_2'] = net
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            endpoints['conv3_3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            endpoints['pool3'] = net

            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            endpoints['conv4_1'] = net
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            endpoints['conv4_2'] = net
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            endpoints['conv4_3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            endpoints['pool4'] = net

            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            endpoints['conv5_1'] = net
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            endpoints['conv5_2'] = net
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            endpoints['conv5_3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            endpoints['pool5'] = net

            dim = 1
            for i in net.shape[1:]:
                dim = dim * int(i)
            net = tf.reshape(net, [batch_size, dim])
            endpoints['reshape'] = net

            net = slim.fully_connected(net, 4096, scope='fc6')
            endpoints['fc6'] = net
            net = slim.dropout(net, dropout, scope='dropout6')
            endpoints['dropout6'] = net

            net = slim.fully_connected(net, 4096, scope='fc7')
            endpoints['fc7'] = net
            net = slim.dropout(net, dropout, scope='dropout7')
            endpoints['dropout7'] = net

            net = slim.fully_connected(
                net, num_classes, activation_fn=None, scope='fc8')
            endpoints['fc8'] = net

            return net, endpoints


def predict(float_images, labels, ids, batch_size, ckpt_dir,
            csv_dict, do_verbose=False):

    with tf.device('/gpu:0'):

        image_batch = tf.map_fn(lambda frame: (frame / 255.0),
                                float_images)

        print('Batch size : ', batch_size)
        logits, endpoints = vgg16(image_batch,
                                  num_classes=params.N_CLASSES,
                                  dropout=1.0,
                                  batch_size=batch_size
                                  )

        logits_argmax = tf.argmax(logits, axis=1)
        softmax_logits = tf.nn.softmax(logits, name=None)
        softmax_logits_argmax = tf.argmax(softmax_logits, axis=1)

        correct_pred = tf.equal(tf.cast(softmax_logits_argmax, tf.int32),
                                tf.cast(labels, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=params.LOG_DEVICE_PLACEMENT))

        restore_file = tf.train.latest_checkpoint(ckpt_dir)
        print(restore_file)
        saver.restore(sess, restore_file)
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        start_time = time.time()

        labels_out,\
            logits_out, softmax_logits_out, \
            logits_argmax_out, accuracy_out,\
            ids_out, labels_out = sess.run([labels,
                                            logits,
                                            softmax_logits,
                                            logits_argmax,
                                            accuracy,
                                            ids,
                                            labels])

        duration = time.time() - start_time

        # ============================================================
        for i in range(batch_size):
            csv_dict['CellID'].append(ids_out[i])
            csv_dict['ImageID'].append('%s.png' % ids_out[i].split('_')[0])
            csv_dict['Labels'].append(labels_out[i])

            for j in range(params.N_CLASSES):
                csv_dict['Logits_%s' % j].append(softmax_logits_out[i, j])
            csv_dict['Predictions'].append(logits_argmax_out[i])

    # ============================================================

    return csv_dict


if __name__ == '__main__':

    TFRECORD_DIR = './DISEASED_SAMPLES_TFRECORD_WHOLE_96/'
    CROPPED_CELL_PATH = './DISEASED_SAMPLES_CROPPEDCELLS_WHOLE_96/'
    SAVE_DIR = './DISEASED_SAMPLES_RESULTS_WHOLE_96/'

    CKPT_DIR = './summary_Round5/'

    files = glob(CROPPED_CELL_PATH + '*/*.png')
    batch_size = len(files)

    csv_dict = {}
    csv_dict['ImageID'] = []
    csv_dict['CellID'] = []
    csv_dict['Labels'] = []
    csv_dict['Predictions'] = []
    for i in range(params.N_CLASSES):
        csv_dict['Logits_%d' % i] = []
    columns = ['ImageID', 'CellID', 'Labels', 'Predictions']
    columns += ['Logits_%d' % i for i in range(params.N_CLASSES)]

    for i in range(16):
        with tf.Graph().as_default():

            images, labels, ids = test_inputs\
                .distorted_inputs([TFRECORD_DIR + 'slide_test.tfrecords'],
                                  batch_size, True)

            csv_dict = predict(images,
                               labels,
                               ids,
                               batch_size,
                               CKPT_DIR,
                               csv_dict)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        df = pd.DataFrame(csv_dict, columns=columns)
        df.to_csv(
            SAVE_DIR + 'predictions_classification.csv', index=True)
