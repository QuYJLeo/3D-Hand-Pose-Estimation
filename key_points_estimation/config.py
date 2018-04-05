import os


TF_CPP_MIN_LOG_LEVEL = '2'

EPSILON = 1.0e-16

PI = 3.141592653589793

(R_TRAIN_RECORD_PATH, R_VAL_RECORD_PATH, S_TRAIN_RECORD_PATH, S_VAL_RECORD_PATH) = [
    os.path.join(os.path.dirname(__file__), '__tfrecord', _) for _ in [
        'train_R.record', 'val_R.record', 'train_S.record', 'val_S.record'
    ]
]

NUM_KEY_POINTS = 21

BATCH_SIZE = 8

CROP_SIZE = 256

MODEL_PATH = '__model/CPM_MPII.ckpt'

# For training
TRAIN_DIR = '__train'
TRAIN_LEARNING_RATE = ([0, 10000, 20000], [1e-4, 1e-5, 1e-6])
TRAIN_PERIOD_SHOW_LOSS = 10
TRAIN_PERIOD_CKPT = 800
TRAIN_PERIOD_SUMMARY = 100
TRAIN_MAX_STEP = 30000
TRAIN_max_to_keep = 100
TRAIN_keep_checkpoint_every_n_hours = 10000

# For evaluating
VAL_DIR = '__eval'
VAL_PERIOD_CHECK = 15
