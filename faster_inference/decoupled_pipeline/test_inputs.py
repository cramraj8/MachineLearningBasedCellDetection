# -*- coding: utf-8 -*-
# @__ramraj__


from __future__ import division, print_function, absolute_import
import tensorflow as tf
import params
import os
import math


FLAGS = tf.app.flags.FLAGS


def distorted_inputs(record_file, batch_size=FLAGS.batch_size, do_test=True):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64),
               'train/id': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(record_file,
                                                    num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.cast(features['train/label'], tf.int32)
    ID = features['train/id']

    # image = tf.reshape(image, [FLAGS.orig_size, FLAGS.orig_size, 3])
    image = tf.reshape(image, [FLAGS.img_size, FLAGS.img_size, 3])

    with tf.name_scope('data_augmentation'):

        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        # Randomly crop a [height, width] section of the image.

        # distorted_image = tf.random_crop(
            # image, [FLAGS.img_size, FLAGS.img_size, 3])
        distorted_image = image

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        distorted_image = tf.image.random_flip_up_down(distorted_image)

        # tf.image.transpose_image ??

        # tf.image.rgb_to_grayscale
        # if random.choice([True, False]):
        # distorted_image = tf.image.rgb_to_grayscale(distorted_image)
        # tf.image.rgb_to_hsv -> done by PLOS

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

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    # num_examples_per_epoch = FLAGS.n_test
    num_examples_per_epoch = batch_size
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    distorted_image.set_shape((FLAGS.img_size, FLAGS.img_size, 3))

    num_preprocess_threads = 16
    image_batch, label_batch, ID_batch = tf.train.shuffle_batch(
        [distorted_image, label, ID],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        allow_smaller_final_batch=True)
    # )

    tf.summary.image('images', image_batch)
    return image_batch, label_batch, ID_batch


def inputs(record_file, batch_size=FLAGS.batch_size, do_test=True):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64),
               'train/id': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(record_file,
                                                    num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.cast(features['train/label'], tf.int32)
    ID = features['train/id']

    # image = tf.reshape(image, [FLAGS.orig_size, FLAGS.orig_size, 3])
    image = tf.reshape(image, [FLAGS.img_size, FLAGS.img_size, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    # num_examples_per_epoch = FLAGS.n_test
    num_examples_per_epoch = batch_size
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    image.set_shape((FLAGS.img_size, FLAGS.img_size, 3))

    num_preprocess_threads = 16
    image_batch, label_batch, ID_batch = tf.train.shuffle_batch(
        [image, label, ID],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        allow_smaller_final_batch=True)
    # )

    tf.summary.image('images', image_batch)
    return image_batch, label_batch, ID_batch


if __name__ == '__main__':

    import sys

    ROUND = sys.argv[1]
    print('\nExperiment ROUND is: %s' % ROUND)

    batch_size = int(sys.argv[2])
    print('\nExperiment batch_size is: %s' % batch_size)

    # ROUND = 1

    images, labels, ids = \
        distorted_inputs(['./TEST_Record_ROUND_%s/slide_test.tfrecords' % ROUND],
                         batch_size, True)

    print(images.shape)
    print(labels.shape)
    print(ids.shape)

    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # img = sess.run(images)
        lbl = sess.run(labels)
        # iidds = sess.run(ids)

        coord.request_stop()
        coord.join(threads)
        sess.close()

    # print(img[0, :, :, :])
    # print(iidds)
    # print(img.shape)
    print(lbl[:5])

    # import matplotlib.pyplot as plt
    # plt.imshow(img[0])
    # plt.show()
    # import cv2
    # cv2.imwrite('delete.png', img[0])
