# -*- coding: utf-8 -*-
# @__ramraj__


import os
import pandas as pd
import numpy as np


COLOR_CODE = ['rgb(0,0,255)', 'rgb(128,0,128)', 'rgb(153, 51, 255)',
              'rgb(153,0,255)', 'rgb(255,0,102)', 'rgb(0,0,153)',
              'rgb(102,0,255)',
              'rgb(255,80,80)', 'rgb(0,102,204)', 'rgb(102,102,153)', 'rgb(0,204,255)',
              'rgb(0,0,0)']


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def do_aggregation(df):

    json_dict = []

    cellids = df['CellID']
    set_cellids = list(set(cellids))

    # Unique Cell ID elements
    output_df = pd.DataFrame()
    softmax_sum = np.asarray([], dtype=np.float32)
    for i in range(len(set_cellids)):
        tmp_cellid = set_cellids[i]
        tmp_pred = df[df['CellID'] == tmp_cellid]['Predictions']
        tmp_softmax = df[df['CellID'] == tmp_cellid].iloc[:, 2:]

        # +++++++++++++++ Aggregated Softmax +++++++++++++++++++++
        tmp_softmax = np.asarray(tmp_softmax)
        tmp_softmax_sum = np.sum(tmp_softmax, axis=0)
        tmp_vote = np.argmax(tmp_softmax_sum)
        tmp_norm_softmax = softmax(tmp_softmax_sum)

        output_df.loc[i, 'CellID'] = tmp_cellid
        output_df.loc[i, 'Predictions'] = int(tmp_vote)

        # ====================================================================

        tmp = tmp_cellid.split('_')
        prob = tmp[0]
        x_c = int(tmp[1])
        y_c = int(tmp[2])
        bbox_h = int(tmp[4])
        bbox_w = int(tmp[3])
        label = tmp[5]
        pred_color = COLOR_CODE[int(tmp_vote)]

        tmp_json_dict = {}
        tmp_json_dict['type'] = "rectangle"
        tmp_json_dict['center'] = [x_c, y_c, 0]
        tmp_json_dict['width'] = bbox_w
        tmp_json_dict['height'] = bbox_h
        tmp_json_dict['rotation'] = 0
        tmp_json_dict['fillColor'] = "rgba(0,0,0,0)"
        tmp_json_dict['lineColor'] = pred_color

        json_dict.append(tmp_json_dict)

    return json_dict


if __name__ == '__main__':

    df = pd.read_csv('./all_classify.csv')

    print('Pandas shape : ', df.shape)
    do_aggregation(df)
