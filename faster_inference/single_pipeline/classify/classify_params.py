# -*- coding: utf-8 -*-
# @__ramraj__


from __future__ import absolute_import, division, print_function
import tensorflow as tf


ROUND = 0
n_train = 6799
n_test = 2076
n_gpus = 4
ROI_SLIDES = ['91316_leica_at2_40x', '93093', '93095', '93090', '93091',
              '93094', '93092', '93096',
              '93098', '93099', '93097', '91315_leica_at2_40x']
TEST_SLIDES = ROI_SLIDES[2 * int(ROUND): 2 * int(ROUND) + 2]


CLASSES = ['blast', 'band_seg', 'metamyelocyte',
           'myelocyte', 'eosinophil', 'promyelocyte',
           'basophil',
           'erythroid', 'lymphocyte', 'monocyte', 'plasma',
           'unknown']
COMPACT_CLASSES = ['myeloid',
                   'erythroid', 'lymphocyte', 'monocyte', 'plasma',
                   'unknown']
MYELOID_CLASSES = ['blast', 'band_seg', 'metamyelocyte',
                   'myelocyte', 'eosinophil', 'promyelocyte',
                   'basophil']


SLIDES = ['93090', '93091', '93092', '93093', '93094',
          '93095', '93096', '93097', '93098', '93099',
          '91315_leica_at2_40x', '91316_leica_at2_40x',
          '93104',
          '101626', '101627', '101633', '101636']


TOWER_NAME = 'tower'


MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100000.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05       # Initial learning rate.


LOG_DEVICE_PLACEMENT = False


# Data & Model file & dir locations
data_dir = './Data106/'
tfrecord_dir = './records96_Round%s/' % ROUND
summary_dir = './summary_Round%s/' % ROUND
pred_dir = './predictions_Round%s/' % ROUND
post_fix = '_EXP_ROUND_%s' % ROUND


# Data Specifications
split = 0.2
orig_size = 106
img_size = 96
display_step = 2


# Model Hyper-parameters
dropout = 0.7
batch_size = 100
test_batch_size = 100
n_epochs = 500
lr = 0.0001

# Other parameters
do_finetune = False
do_fixed_lr = True
# Derived Parameters
N_CLASSES = len(CLASSES)


N_TRAIN_BATCHES = int(n_train / batch_size)
max_train_steps = int(N_TRAIN_BATCHES / n_gpus) * n_epochs


N_TEST_BATCHES = int(n_test / test_batch_size)
max_test_steps = N_TEST_BATCHES * n_epochs
