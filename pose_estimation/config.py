TF_CPP_MIN_LOG_LEVEL = '2'

NUM_KEY_POINTS = 21

BATCH_SIZE = 8

R_TRAIN_RECORD_PATH = '../key_points_estimation/__tfrecord/train_R.record'
R_VAL_RECORD_PATH = '../key_points_estimation/__tfrecord/val_R.record'
S_TRAIN_RECORD_PATH = '../key_points_estimation/__tfrecord/train_S.record'
S_VAL_RECORD_PATH = '../key_points_estimation/__tfrecord/val_S.record'

# For training
TRAIN_DIR = '__train'
TRAIN_LEARNING_RATE = ([0, 60000], [1e-5, 1e-6])
TRAIN_PERIOD_SHOW_LOSS = 20
TRAIN_PERIOD_CKPT = 2000
TRAIN_PERIOD_SUMMARY = 400
TRAIN_MAX_STEP = 80000
TRAIN_max_to_keep = 100
TRAIN_keep_checkpoint_every_n_hours = 10000

# For evaluating
VAL_DIR = '__eval'
VAL_PERIOD_CHECK = 15
