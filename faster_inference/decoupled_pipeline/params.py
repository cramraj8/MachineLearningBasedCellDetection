# -*- coding: utf-8 -*-
# @__ramraj__


from __future__ import absolute_import, division, print_function
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


ROUND = 5
tf.app.flags.DEFINE_integer('n_train', 7764,
                            """Number of examples in train dataset.""")
tf.app.flags.DEFINE_integer('n_test', 1111,
                            """Number of examples in test dataset.""")
tf.app.flags.DEFINE_integer('n_gpus', 2,
                            """Number of GPUs in use.""")
ROI_SLIDES = ['91316_leica_at2_40x', '93093', '93095', '93090', '93091',
              '93094', '93092', '93096',
              '93098', '93099', '93097', '91315_leica_at2_40x']
TEST_SLIDES = ROI_SLIDES[2 * int(ROUND): 2 * int(ROUND) + 2]

"""
Things to change:
- round
- n_train
- n_test
- n_gpu
"""


# All classes present
# CLASSES = ['unknown', 'band_seg', 'blast', 'eosinophil', 'erythroid', 'lymphocyte',
#            'metamyelocyte', 'monocyte', 'myelocyte', 'plasma', 'promyelocyte',
#            'basophil', 'megakaryocyte']


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
tf.app.flags.DEFINE_string('data_dir', './Data106/',
                           """Path to the image data directory.""")
tf.app.flags.DEFINE_string('tfrecord_dir', './records96_Round%s/' % ROUND,
                           """Path to the TF-Records data directory.""")
tf.app.flags.DEFINE_string('summary_dir', './summary_Round%s/' % ROUND,
                           """Path to the TF Summary directory.""")
tf.app.flags.DEFINE_string('pred_dir', './predictions_Round%s/' % ROUND,
                           """Path to the predictions directory.""")
tf.app.flags.DEFINE_string('post_fix', '_EXP_ROUND_%s' % ROUND,
                           """Post fix name for different runs of same experiment.""")


# Data Specifications
tf.app.flags.DEFINE_float('split', 0.2,
                          """Ration for train test split.""")
tf.app.flags.DEFINE_integer('seed', 676,
                            """Random seed to shuffle data loading.""")
tf.app.flags.DEFINE_integer('orig_size', 106,
                            """Original image size converted from TFRecord.""")
tf.app.flags.DEFINE_integer('img_size', 96,
                            """Random crop image size.""")
tf.app.flags.DEFINE_integer('display_step', 2,
                            """Number of steps to skip while displaying the metrics.""")


# Model Hyper-parameters
tf.app.flags.DEFINE_float('dropout', 0.7,
                          """Dropout for regularization.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Batchsize during training.""")
tf.app.flags.DEFINE_integer('test_batch_size', 100,
                            """Batchsize during testing.""")
tf.app.flags.DEFINE_integer('n_epochs', 500,
                            """Total number of epochs.""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                          """Fixed learning rate.""")


# Other parameters
tf.app.flags.DEFINE_boolean('do_finetune', False,
                            """Do finetuning by loading ckpts.""")
tf.app.flags.DEFINE_boolean('do_fixed_lr', True,
                            """Do Fixed learning rate.""")

# Derived Parameters
N_CLASSES = len(CLASSES)


N_TRAIN_BATCHES = int(FLAGS.n_train / FLAGS.batch_size)
max_train_steps = int(N_TRAIN_BATCHES / FLAGS.n_gpus) * FLAGS.n_epochs


N_TEST_BATCHES = int(FLAGS.n_test / FLAGS.test_batch_size)
max_test_steps = N_TEST_BATCHES * FLAGS.n_epochs

