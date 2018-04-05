import os
import pickle
import glob
import subprocess
import json

import PIL
import numpy as np
import cv2
import tensorflow as tf

from fingerspelling_recognition.config import *

from utils.ProgressMsgDisplayer import ProgressMsgDisplayer

from test.run import detect_hand, estimate_key_points, estimate_pose, draw_hand


Example = dict


def extract_mid_frame(video_file):
    img_file = os.path.join(TMP_DIR, 'img', os.path.basename(os.path.splitext(video_file)[0])) + '.bmp'

    # Get some information about `video_file`
    cmd = ['ffprobe', '-i', video_file, '-show_format', '-print_format', 'json', '-hide_banner']
    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    out = run.stdout.decode('UTF-8')
    out = json.loads(out)

    start_time = float(out['format']['start_time'])
    duration = float(out['format']['duration'])

    # Extract middle frame
    cmd = ['ffmpeg', '-ss', str(start_time + duration / 2), '-i', video_file,
           '-y', '-frames:v', '1', img_file, '-hide_banner']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=True)

    return img_file


def _func_1(data_dir):
    pickle_path = os.path.join(TMP_DIR, '_func_1.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as ifs:
            return pickle.load(ifs)

    labels = []
    for _ in LABELS_ALL:
        if _ not in LABELS_EXCLUDED:
            labels.append(_)
    labels_map = dict(enumerate(labels))
    labels_map_r = dict(zip(labels_map.values(), labels_map.keys()))

    print('Gathering video filepaths ...... ', end='')

    examples_train, examples_val = [], []
    data_dir = os.path.join(data_dir, 'video')
    for _ in os.listdir(data_dir):
        video_dir = os.path.join(data_dir, _)
        assert os.path.isdir(video_dir)

        label_id = int(_) - 1
        assert 0 <= label_id <= 34

        label = LABELS_ALL[label_id]
        label_id = labels_map_r.get(label)
        if label_id is None:
            continue

        _list = []
        for video_file in glob.glob(os.path.join(video_dir, '*cam1.mpg')):
            _list.append(Example(video_file=video_file, label_id=label_id))

        np.random.shuffle(_list)

        num_train = int(len(_list) * PERCENTAGE_TRAIN)
        examples_train.extend(_list[:num_train])
        examples_val.extend(_list[num_train:])

    print('Done')

    examples = examples_train + examples_val

    print('')
    print('Extracting middle frames ......')
    with ProgressMsgDisplayer() as progress_msg_displayer:
        for i in range(len(examples)):
            example = examples[i]
            progress_msg_displayer.update('%04d/%04d %s' % (i + 1,
                                                            len(examples),
                                                            os.path.basename(example['video_file'])))
            example['img_file'] = extract_mid_frame(example['video_file'])

    with open(pickle_path, 'wb') as ofs:
        pickle.dump((labels_map, examples_train, examples_val), ofs)

    return labels_map, examples_train, examples_val


def extract_features(img_file, debug=True):
    img = np.array(PIL.Image.open(img_file))

    boxes, scores = detect_hand(img)
    _ = np.argmax(scores, axis=0)
    box, score = boxes[_], scores[_]

    img_cr, score_maps, uv_cr = estimate_key_points(img, box)
    if debug:
        img_uv_cr = np.copy(img_cr[:, :, ::-1])
        draw_hand(img_uv_cr, uv_cr)

        img_uv_cr_file = os.path.join(TMP_DIR, 'img_uv_cr',
                                      os.path.basename(os.path.splitext(img_file)[0])) + '.jpg'
        cv2.imwrite(img_uv_cr_file, img_uv_cr)

    is_right_hand = True
    _, _, xyz_rel = estimate_pose(score_maps, is_right_hand)

    return np.reshape(xyz_rel, [-1])


def _func_2(data_dir):
    pickle_path = os.path.join(TMP_DIR, '_func_2.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as ifs:
            return pickle.load(ifs)

    labels_map, examples_train, examples_val = _func_1(data_dir)
    examples = examples_train + examples_val

    print('')
    print('Extracting features ......')
    with ProgressMsgDisplayer() as progress_msg_displayer:
        for i in range(len(examples)):
            example = examples[i]
            progress_msg_displayer.update('%04d/%04d %s' % (i + 1,
                                                            len(examples),
                                                            os.path.basename(example['img_file'])))
            example['features'] = extract_features(example['img_file'])

    with open(pickle_path, 'wb') as ofs:
        pickle.dump((labels_map, examples_train, examples_val), ofs)

    return labels_map, examples_train, examples_val


def get_examples(data_dir):
    return _func_2(data_dir)


def create_tf_example(example):
    img_file = example['img_file']
    features = example['features']
    label_id = example['label_id']

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'img_file': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_file.encode('utf-8')])),
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'label_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
    }))

    return tf_example


def gen_tfrecord(tfrecord_path, examples):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    print('')
    print('Generating %s ......' % tfrecord_path)

    with ProgressMsgDisplayer() as progress_msg_displayer:
        i = 1
        for example in examples:
            progress_msg_displayer.update('%04d/%04d' % (i, len(examples)))

            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
            i += 1

    writer.close()


def exclude_unwanted_examples(examples_train, examples_val):
    img_uv_cr_dir = os.path.join(TMP_DIR, 'img_uv_cr')
    while True:
        s = input('Delete unwanted examples in "%s", then type "go" to continue: ' % img_uv_cr_dir)
        if s == 'go':
            break

    included_files = glob.glob(os.path.join(img_uv_cr_dir, '*.jpg'))
    included_files = [os.path.basename(os.path.splitext(_)[0]) for _ in included_files]
    included_files = set(included_files)

    files_train = [os.path.basename(os.path.splitext(_['img_file'])[0]) for _ in examples_train]
    files_val = [os.path.basename(os.path.splitext(_['img_file'])[0]) for _ in examples_val]

    file2example = dict(zip(files_train + files_val, examples_train + examples_val))

    files_train = list(set(files_train) & included_files)
    files_val = list(set(files_val) & included_files)

    examples_train = [file2example[_] for _ in files_train]
    examples_val = [file2example[_] for _ in files_val]

    assert len(examples_train) + len(examples_val) == len(included_files)

    return examples_train, examples_val


def main():
    labels_map, examples_train, examples_val = get_examples(DATA_DIR)

    examples_train, examples_val = exclude_unwanted_examples(examples_train, examples_val)

    for _ in [examples_train, examples_val]:
        np.random.shuffle(_)

    for tfrecord_path, examples in [(TRAIN_RECORD_PATH, examples_train),
                                    (VAL_RECORD_PATH, examples_val)]:
        gen_tfrecord(tfrecord_path, examples)


if __name__ == '__main__':
    main()
