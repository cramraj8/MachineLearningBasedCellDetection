# -*- coding: utf-8 -*-
# @__ramraj__


import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import sys


sys.path.append('../Detection/luminoth')

from luminoth.vis import vis_objects
from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork


import cell_croppings


CONFIG = './sample_config.yml'

CROP_CELLS_DIR = './DISEASED_SAMPLES_CROPPEDCELLS_WHOLE_96/objectness/'
if not os.path.exists(CROP_CELLS_DIR):
    os.makedirs(CROP_CELLS_DIR)


MAX_DET = 800
MIN_PROB = 0.1


def main(input_image):

    tf.reset_default_graph()

    # **********************************************************************

    with tf.gfile.Open(input_image, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
        except (tf.errors.OutOfRangeError, OSError) as e:
            print 'Exception!'

    # **********************************************************************

    config = get_config(CONFIG)
    config.model.rcnn.proposals.total_max_detections = MAX_DET
    config.model.rcnn.proposals.min_prob_threshold = MIN_PROB

    network = PredictorNetwork(config)
    objects = network.predict_image(image)

    print '************************************** Num of Objects : ', len(objects)
    # **********************************************************************

    ref_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    pref_list, images, ids = cell_croppings.load_json(
        ref_image,
        preds=objects)
    print '************************************** Num of Cropped Objects : ', np.shape(ids)

    # **********************************************************************

    if not os.path.exists(CROP_CELLS_DIR):
        os.makedirs(CROP_CELLS_DIR)

    for i in range(ids.shape[0]):

        # SRC_DIR / IMAGE_ID _ COORDS . LABEL

        dst_filename = '%s_%s_%s.png' % (
            os.path.basename(input_image)[:-4], ids[i], i)

        dst_img_name = os.path.join(CROP_CELLS_DIR, dst_filename)

        cv2.imwrite(dst_img_name, images[i, ...])


if __name__ == '__main__':
    path = '../fulldata'
    files = os.listdir(path)

    for file in files:
        main(os.path.join(path, file))
