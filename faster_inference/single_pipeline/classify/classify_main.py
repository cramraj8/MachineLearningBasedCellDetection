# -*- coding: utf-8 -*-
# @__ramraj__


from __future__ import division, print_function, absolute_import
import tensorflow as tf
import pandas as pd
import tensorflow.contrib.slim as slim
import cv2
import math

from Classify import classify_params
from Classify import classify_agg


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


def predict(images_in, ids, ckpt_file, do_verbose=False):

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # +++++++++++++++++++++++++ Loaded Numpy Data +++++++++++++++++++++++++

        batch_size = images_in.shape[0]

        # +++++++++++++++++++++ Initialize Placeholders +++++++++++++++++++++++

        images = tf.placeholder(tf.float32, shape=[batch_size,
                                                   classify_params.orig_size, classify_params.orig_size,
                                                   3])
        IDs = tf.placeholder(tf.string, [batch_size, ])

        # ++++++++++++++++++++++++ Do Augmentation ++++++++++++++++++++++++++++

        images = tf.reshape(
            images, [batch_size, classify_params.orig_size, classify_params.orig_size, 3])

        with tf.name_scope('data_augmentation'):

            # Randomly crop a [height, width] section of the image.
            distorted_image = tf.random_crop(images,
                                             [batch_size,
                                              classify_params.img_size, classify_params.img_size, 3])
            # Randomly flip the image horizontally.
            distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                                        distorted_image)
            distorted_image = tf.map_fn(lambda img: tf.image.random_flip_up_down(img),
                                        distorted_image)

            # Random Rotation among [0, 90, 180, 270]
            rand_num = tf.random_uniform([1], 0, 4, tf.float32)
            rand_num = tf.cast(rand_num, tf.int32)
            rand_angle = tf.cast(rand_num, tf.float32) * math.pi / 2
            distorted_image = tf.contrib.image.rotate(distorted_image,
                                                      rand_angle,
                                                      interpolation='BILINEAR')

            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=0.25)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.9, upper=1.4)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        image_batch = tf.map_fn(lambda frame: (frame / 255.0),
                                distorted_image)

        logits, endpoints = vgg16(image_batch,
                                  num_classes=classify_params.N_CLASSES,
                                  dropout=1.0,
                                  batch_size=batch_size
                                  )

        logits_argmax = tf.argmax(logits, axis=1)
        softmax_logits = tf.nn.softmax(logits, name=None)
        softmax_logits_argmax = tf.argmax(softmax_logits, axis=1)

        saver = tf.train.Saver()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=config)

        restore_file = tf.train.latest_checkpoint(ckpt_file)
        saver.restore(sess, restore_file)

        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        csv_dict = {}
        csv_dict['CellID'] = []
        csv_dict['Predictions'] = []
        logits_col = ['Softmax_%s' %
                      l for l in range(classify_params.N_CLASSES)]
        for k in range(classify_params.N_CLASSES):
            csv_dict[logits_col[k]] = []

        print('--->> Classification Graph Model Created')

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(16):

            softmax_logits_argmax_out,\
                softmax_logits_out, \
                IDs_out = sess.run([softmax_logits_argmax,
                                    softmax_logits,
                                    IDs],
                                   feed_dict={images: images_in,
                                              IDs: ids})

            for j in range(softmax_logits_argmax_out.shape[0]):
                csv_dict['CellID'].append(IDs_out[j])
                csv_dict['Predictions'].append(softmax_logits_argmax_out[j])
                for k in range(classify_params.N_CLASSES):
                    csv_dict['Softmax_%s' % k].append(softmax_logits_out[j][k])

            print('Test Augmentation Round: %s Calculated...' % i)

        # ============================================================
        print('--->> Starting Softmax Aggregation')
        df = pd.DataFrame(csv_dict,
                          columns=['CellID', 'Predictions'] + logits_col)

        # ============================================================
        classification_json = classify_agg.do_aggregation(df)
        print('--->> Finished Softmax Aggregation')

        return classification_json


if __name__ == '__main__':
    pass
