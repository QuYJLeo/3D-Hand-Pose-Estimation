
TF_CPP_MIN_LOG_LEVEL = '2'

DATA_DIR = '../data/T'

TMP_DIR = '__tmp/'

LABELS_ALL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'AE', 'OE', 'UE', 'SCH', 'One', 'Two', 'Three', 'Four', 'Five']

LABELS_EXCLUDED = ['J', 'Z', 'AE', 'OE', 'UE']

TRAIN_RECORD_PATH = '__tfrecord/train.record'
VAL_RECORD_PATH = '__tfrecord/val.record'

NUM_CLASSES = len(LABELS_ALL) - len(LABELS_EXCLUDED)

BATCH_SIZE = 8

PERCENTAGE_TRAIN = 0.8

# For training
TRAIN_DIR = '__train'
TRAIN_LEARNING_RATE = ([0, 20000, 30000], [1e-4, 1e-5, 5e-6])
TRAIN_PERIOD_SHOW_LOSS = 200
TRAIN_PERIOD_CKPT = 2000
TRAIN_PERIOD_SUMMARY = 2000
TRAIN_MAX_STEP = 40000
TRAIN_max_to_keep = 100
TRAIN_keep_checkpoint_every_n_hours = 10000

# For evaluating
VAL_DIR = '__eval'
VAL_PERIOD_CHECK = 15
