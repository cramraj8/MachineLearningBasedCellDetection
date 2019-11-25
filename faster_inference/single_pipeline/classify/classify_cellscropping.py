# -*- coding: utf-8 -*-
# @__ramraj__


import numpy as np
import os
import cv2
import tensorflow as tf


# PAD_SIZE = 32
ORIG_SIZE = 106
ORIG_HALF_SIZE = int(ORIG_SIZE / 2)


def load_json(roi_image, preds):
    """
    This function will load each predicted JSON files.
    Read each predicted object's 4 coordinates and its labels with confidence score/probability to be an object.
    Return them as a 2-dim list.

    """

    images = []
    ids = []

    # ++++++++++++++++++++++++++ fILTERING ++++++++++++++++++++++++++
    # Remove the cells, which are close to borders within the PAD_SIZE pixels
    pred_filtered_list = []
    for obj_idx, obj in enumerate(preds):

        x1 = obj['bbox'][0]
        y1 = obj['bbox'][1]
        x2 = obj['bbox'][2]
        y2 = obj['bbox'][3]

        # **********************************************************************

        prob = obj['prob']
        label = obj['label']
        bbox_h = abs(int(x2) - int(x1))
        bbox_w = abs(int(y2) - int(y1))

        # **********************************************************************

        # Get center bbox coordintes
        x_cent = int((x1 + x2) / 2)
        y_cent = int((y1 + y2) / 2)

        value = [obj['label'],      # object class - objectness
                 x1,                # x1 coordinate
                 y1,                # y1 coordinate
                 x2,                # x2 coordinate
                 y2,                # y2 coordinate
                 obj['prob']]       # Confidence scor of this detected objectness

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_x = x_cent - ORIG_HALF_SIZE
        end_x = x_cent + ORIG_HALF_SIZE
        start_y = y_cent - ORIG_HALF_SIZE
        end_y = y_cent + ORIG_HALF_SIZE
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if start_x <= 0:
            start_x = 0

        if end_x > roi_image.shape[1]:
            end_x = roi_image.shape[1]

        if start_y <= 0:
            start_y = 0

        if end_y > roi_image.shape[0]:
            end_y = roi_image.shape[0]

        # ++++++++++++++++++++++++++++++++ padding ++++++++++++++++++++++++++++
        pad_img = np.zeros((ORIG_SIZE, ORIG_SIZE, 3))
        tmp_crop = roi_image[start_y: end_y, start_x: end_x, :]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for i in range(tmp_crop.shape[0]):
            for j in range(tmp_crop.shape[1]):
                pad_img[i, j] = tmp_crop[i, j]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if pad_img.shape != (ORIG_SIZE, ORIG_SIZE, 3):
            print 'Non expeceted crop image shape !!!', pad_img.shape

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cell_img = pad_img

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++ Remove all black crops++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if np.all(cell_img == 0):
            continue

        images.append(cell_img)
        ids.append('%s_%s_%s_%s_%s_%s' % (prob,
                                          x_cent, y_cent, bbox_h, bbox_w,
                                          label))
        pred_filtered_list.append(value)

    images = np.array(images, dtype=np.float32)
    ids = np.array(ids)
    return pred_filtered_list, images, ids


if __name__ == '__main__':

    pass
