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


sys.path.append('./luminoth')

from luminoth.vis import vis_objects
from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork


from classify import classify_main
from classify import classify_cellscropping


CONFIG = './luminoth/examples/sample_config.yml'
CKPT = './classify/ckpt/'


MAX_DET = 800
MIN_PROB = 0.1

OUTPUT_JSON_DIR = 'classification_json_outputs'


def main(input_image):

    # *************************************************************************

    with tf.gfile.Open(input_image, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
        except (tf.errors.OutOfRangeError, OSError) as e:
            print 'Exception!'

    # *************************************************************************

    config = get_config(CONFIG)
    config.model.rcnn.proposals.total_max_detections = MAX_DET
    config.model.rcnn.proposals.min_prob_threshold = MIN_PROB

    network = PredictorNetwork(config)
    objects = network.predict_image(image)

    print '************************* Num of Objects : ', len(objects)
    # *************************************************************************

    ref_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    pref_list, images, ids = classify_cellscropping.load_json(
        ref_image,
        preds=objects)
    print '************************* Num of Cropped Objects : ', np.shape(ids)

    # *************************************************************************

    final_json = classify_main.predict(images, ids, CKPT)
    output_json_path = os.path.join(
        OUTPUT_JSON_DIR, '%s.json' % input_image.split('/')[-1][:-4])
    with open(output_json_path, 'w') as fp:
        json.dump(final_json, fp)


if __name__ == '__main__':
    path = 'fulldata'
    files = os.listdir(path)

    for file in files:
        main(os.path.join(path, file))
