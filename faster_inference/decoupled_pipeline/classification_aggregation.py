# -*- coding: utf-8 -*-
# @__ramraj__


import os
import pandas as pd
from statistics import mode
import numpy as np
from sklearn.metrics import accuracy_score
import params


PREDICTION_CSV_RESULTS = './DISEASED_SAMPLES_RESULTS_WHOLE_96/predictions_classification.csv'
PREDICTION_AGG_CSV_RESULTS = './DISEASED_SAMPLES_RESULTS_WHOLE_96/predictions_agg_classification.csv'


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def create_csv_file():
    df = pd.read_csv(PREDICTION_CSV_RESULTS)

    cellids = df['CellID']
    set_cellids = list(set(cellids))
    print cellids.shape

    output_df = pd.DataFrame()

    softmax_sum = np.asarray([], dtype=np.float32)
    for i in range(len(set_cellids)):
        tmp_cellid = set_cellids[i]

        tmp_lbl = df[df['CellID'] == tmp_cellid][
            'Labels'].iloc[0]  # take either of 16 samples label
        tmp_imageid = df[df['CellID'] == tmp_cellid][
            'ImageID'].iloc[0]  # take either of 16 samples label
        tmp_pred = df[df['CellID'] == tmp_cellid]['Predictions']
        tmp_softmax = df[df['CellID'] == tmp_cellid].iloc[:, 5:]

        # +++++++++++++++++++++ Method 1 - mode voting +++++++++++++++++++++
        # tmp_softmax = np.asarray(tmp_softmax)
        # tmp_argmax = np.argmax(tmp_softmax, axis=1)
        # tmp_vote = mode(tmp_argmax)

        # output_df.loc[i, 'CellID'] = tmp_cellid
        # output_df.loc[i, 'Labels'] = tmp_lbl
        # output_df.loc[i, 'Predictions'] = tmp_vote

        # output_df.to_csv('Output_TEST_TFRecord.csv', index=True)
        # +++++++++++++++ Method 2 - Aggregated Softmax +++++++++++++++++++++
        tmp_softmax = np.asarray(tmp_softmax)
        tmp_softmax_sum = np.sum(tmp_softmax, axis=0)
        tmp_vote = np.argmax(tmp_softmax_sum)

        tmp_norm_softmax = softmax(tmp_softmax_sum)

        output_df.loc[i, 'CellID'] = tmp_cellid
        output_df.loc[i, 'ImageID'] = tmp_imageid
        output_df.loc[i, 'Labels'] = tmp_lbl
        output_df.loc[i, 'Predictions'] = tmp_vote
        for j in range(params.N_CLASSES):
            output_df.loc[i, 'Softmax_%s' % j] = tmp_norm_softmax[j]

        output_df.to_csv(PREDICTION_AGG_CSV_RESULTS, index=True)


# def evaluate_metrics(r):
#     df = pd.read_csv(
#         './Predict_TESTAug_8/TestAug_Predictions_ROUND_%s.csv' % r)
#     accurcy_val = accuracy_score(df['Labels'], df['Predictions'])
#     print accurcy_val

#     return accurcy_val


f = open("JustRaw_Accuracy_TFRecordTesting.txt", "w")

create_csv_file()
# acc = evaluate_metrics(r)
# f.write('%s\t\t' % r)
# f.write('%s' % acc)
# f.write('\n')

f.close()
