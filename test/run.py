import os

import cv2
import PIL
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from key_points_estimation.config import CROP_SIZE, NUM_KEY_POINTS
from key_points_estimation.KeyPointsNet import KeyPointsNet
from key_points_estimation.eval import get_key_points_from_score_maps

from pose_estimation.PoseNet import PoseNet

from utils.misc import restore_weights

from test.config import config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DIR = '../__backup/test/run'


def load_frozen_graph(frozen_pb_path):
    g = tf.Graph()
    with g.as_default():
        graph_def = tf.GraphDef()
        with open(frozen_pb_path, 'rb') as ifs:
            graph_def.ParseFromString(ifs.read())
            tf.import_graph_def(graph_def, name='')
        return g


def detect_hand(img, *, g=None):
    """
    Return boxes in the form of (y_min/H, x_min/W, y_max/H, x_max/W), and their scores
    """

    _ = config['hand_detection']
    frozen_pb_path = _['frozen_pb_path']

    if g is None:
        g = load_frozen_graph(frozen_pb_path)

    with g.as_default():
        with tf.Session(graph=g) as sess:
            image_tensor = g.get_tensor_by_name('image_tensor:0')
            detection_boxes = g.get_tensor_by_name('detection_boxes:0')
            detection_scores = g.get_tensor_by_name('detection_scores:0')

            img = np.expand_dims(img, axis=0)
            boxes, scores = sess.run([detection_boxes, detection_scores], feed_dict={image_tensor: img})
            return [np.squeeze(_) for _ in [boxes, scores]]


def estimate_key_points(img, box, *, g=None):
    _ = config['key_points_estimation']
    ckpt_path = _['ckpt_path']
    frozen_pb_path = ckpt_path + '.frozen.pb'

    # adjust `box` into a square
    img_H, img_W, _ = img.shape
    box_y_min, box_x_min, box_y_max, box_x_max = box * np.array([img_H, img_W, img_H, img_W])
    box_center_x, box_center_y = (box_x_max + box_x_min) / 2, (box_y_max + box_y_min) / 2
    box_size = max(box_x_max - box_x_min, box_y_max - box_y_min) * 1.1
    box_y_min, box_x_min, box_y_max, box_x_max = [box_center_y, box_center_x, box_center_y, box_center_x] + \
                                                 np.array([-box_size / 2, -box_size / 2, box_size / 2, box_size / 2])
    box = [box_y_min, box_x_min, box_y_max, box_x_max] / np.array([img_H, img_W, img_H, img_W])

    def _load_graph():
        if not os.path.exists(frozen_pb_path):
            g = tf.Graph()
            with g.as_default():
                # build the graph
                img = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='img')
                box = tf.placeholder(tf.float32, shape=(None, 4), name='box')
                img_cr = tf.image.crop_and_resize(img, box, [0], [CROP_SIZE, CROP_SIZE], name='img_cr')

                key_points_net = KeyPointsNet(img_cr / 255.0 - 0.5)
                score_maps = tf.identity(key_points_net['score_maps_list'][-1], name='score_maps')
                uv_cr = tf.identity(get_key_points_from_score_maps(score_maps)[0], name='uv_cr')

                with tf.Session(graph=g) as sess:
                    # restore weights
                    restore_rules = [
                        (None, r'^KeyPointsNet/.*/bias(:\d)?$', None, None),
                        (None, r'^KeyPointsNet/.*/kernel(:\d)?$', None, None)
                    ]
                    restore_weights(sess, ckpt_path, restore_rules)

                    # freeze the graph
                    frozen_graph = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, ['img_cr', 'score_maps', 'uv_cr']
                    )

                    with open(frozen_pb_path, 'wb') as ofs:
                        ofs.write(frozen_graph.SerializeToString())

        return load_frozen_graph(frozen_pb_path)

    if g is None:
        g = _load_graph()

    with g.as_default():
        with tf.Session(graph=g) as sess:
            img_t = g.get_tensor_by_name('img:0')
            box_t = g.get_tensor_by_name('box:0')
            img_cr_t = g.get_tensor_by_name('img_cr:0')
            score_maps_t = g.get_tensor_by_name('score_maps:0')
            uv_cr_t = g.get_tensor_by_name('uv_cr:0')

            img, box = [np.expand_dims(_, axis=0) for _ in [img, box]]
            img_cr, score_maps, uv_cr = sess.run([img_cr_t, score_maps_t, uv_cr_t],
                                                 feed_dict={img_t: img, box_t: box})
            return [np.squeeze(_) for _ in [img_cr.astype(np.uint8), score_maps, uv_cr]]


def estimate_pose(score_maps, is_right_hand, *, g=None):
    _ = config['pose_estimation']
    ckpt_path = _['ckpt_path']
    frozen_pb_path = ckpt_path + '.frozen.pb'
    is_right_hand_vec = np.array([0, 1.0]) if is_right_hand else np.array([1.0, 0])

    def _load_graph():
        if not os.path.exists(frozen_pb_path):
            g = tf.Graph()
            with g.as_default():
                # build the graph
                score_maps = tf.placeholder(tf.float32,
                                            shape=(None, CROP_SIZE, CROP_SIZE, NUM_KEY_POINTS),
                                            name='score_maps')
                is_right_hand_vec = tf.placeholder(tf.float32,
                                                   shape=(None, 2),
                                                   name='is_right_hand_vec')

                score_maps_pooled = tf.nn.avg_pool(score_maps,
                                                   ksize=[1, 8, 8, 1],
                                                   strides=[1, 8, 8, 1],
                                                   padding='SAME')
                pose_net = PoseNet(score_maps_pooled, is_right_hand_vec, training=tf.constant(False))
                xyz_cano = tf.identity(pose_net['xyz_cano'], name='xyz_cano')
                rot_inv = tf.identity(pose_net['rot_inv'], name='rot_inv')
                xyz_rel = tf.identity(pose_net['xyz_rel'], name='xyz_rel')

                with tf.Session(graph=g) as sess:
                    # restore weights
                    tf.train.Saver().restore(sess, ckpt_path)

                    # freeze the graph
                    frozen_graph = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, ['xyz_cano', 'rot_inv', 'xyz_rel']
                    )

                    with open(frozen_pb_path, 'wb') as ofs:
                        ofs.write(frozen_graph.SerializeToString())

        return load_frozen_graph(frozen_pb_path)

    if g is None:
        g = _load_graph()

    with g.as_default():
        with tf.Session(graph=g) as sess:
            score_maps_t = g.get_tensor_by_name('score_maps:0')
            is_right_hand_vec_t = g.get_tensor_by_name('is_right_hand_vec:0')
            xyz_cano_t = g.get_tensor_by_name('xyz_cano:0')
            rot_inv_t = g.get_tensor_by_name('rot_inv:0')
            xyz_rel_t = g.get_tensor_by_name('xyz_rel:0')

            score_maps, is_right_hand_vec = [np.expand_dims(_, axis=0) for _ in [score_maps, is_right_hand_vec]]
            xyz_cano, rot_inv, xyz_rel = sess.run(
                [xyz_cano_t, rot_inv_t, xyz_rel_t],
                feed_dict={
                    score_maps_t: score_maps,
                    is_right_hand_vec_t: is_right_hand_vec
                })
            return [np.squeeze(_) for _ in [xyz_cano, rot_inv, xyz_rel]]


def gen_colors(*, n=NUM_KEY_POINTS - 1, fmt='opencv', h_range=[0, 360.0], s_range=[0.8, 0.8], v_range=[0.8, 0.8]):
    assert fmt in ['opencv', 'matplotlib']

    for i in range(1, n + 1):
        hsv = list(map(lambda _: _[0] + float(i) / n * (_[1] - _[0]), [h_range, s_range, v_range]))
        rgb = cv2.cvtColor(np.array([[hsv]], dtype=np.float32), cv2.COLOR_HSV2RGB)[0][0]
        yield tuple(map(int, rgb[::-1] * 255)) if fmt == 'opencv' else rgb


def gen_bones(pts):
    bones = [
        (0, 4), (4, 3), (3, 2), (2, 1),
        (0, 8), (8, 7), (7, 6), (6, 5),
        (0, 12), (12, 11), (11, 10), (10, 9),
        (0, 16), (16, 15), (15, 14), (14, 13),
        (0, 20), (20, 19), (19, 18), (18, 17)
    ]

    for bone in bones:
        yield tuple(pts[bone[0]]), tuple(pts[bone[1]])


def draw_hand(img, uv):
    for (pt1, pt2), color in zip(gen_bones(uv), gen_colors()):
        pt1, pt2 = [(int(_[0]), int(_[1])) for _ in (pt1, pt2)]
        cv2.line(img, pt1, pt2, color, lineType=cv2.LINE_AA)


def plot_hand_3d(axes, xyz_rel):
    for (pt1, pt2), color in zip(gen_bones(xyz_rel), gen_colors(fmt='matplotlib')):
        _ = np.stack([pt1, pt2])
        axes.plot(_[:, 0], _[:, 1], _[:, 2], color=color, linewidth=1)

    # aligns the 3d coord with the camera view
    axes.view_init(azim=-90.0, elev=-90.0)
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    axes.set_zlim([-3, 3])


def get_save_func(img_file):
    def _save(img, *, i=[0]):
        i[0] += 1
        cv2.imwrite('%s.%02d.jpg' % (img_file, i[0]), img)
    return _save


def get_fig_img(fig):
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def cvt_float_img(img):
    assert len(img.shape) in [2, 3, 4]

    axis = (0, 1) if len(img.shape) < 4 else (1, 2)
    _min = np.min(img, axis=axis, keepdims=True)
    _max = np.max(img, axis=axis, keepdims=True)
    return ((img - _min) / (_max - _min) * 255.0).astype(np.uint8)


def run(img_file_list, is_right_hand_list, *, fig=None, axes=None):
    if fig is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

    cv2.namedWindow('boxes')
    cv2.namedWindow('img_cr')
    cv2.namedWindow('uv_cr')
    cv2.namedWindow('score_map')
    cv2.namedWindow('xyz_rel')

    for img_file, is_right_hand in zip(img_file_list, is_right_hand_list):
        img_file = os.path.join(DIR, img_file)

        print('%s:' % img_file)

        _save = get_save_func(img_file)

        img = np.array(PIL.Image.open(img_file))
        img_H, img_W, _ = img.shape

        boxes, scores = detect_hand(img)
        _ = np.argmax(scores, axis=0)
        box, score = boxes[_], scores[_]

        img_boxes = np.copy(img[:, :, ::-1])
        pt1 = (int(box[1] * img_W), int(box[0] * img_H))
        pt2 = (int(box[3] * img_W), int(box[2] * img_H))
        cv2.rectangle(img_boxes, pt1, pt2, (255, 255, 255), 2)
        cv2.imshow('boxes', img_boxes)

        if score < config['hand_detection']['th2']:
            print('\tNo valid hand detected')

        img_cr, score_maps, uv_cr = estimate_key_points(img, box)
        img_cr = np.copy(img_cr[:, :, ::-1])
        cv2.imshow('img_cr', img_cr)

        img_uv_cr = np.copy(img_cr)
        draw_hand(img_uv_cr, uv_cr)
        cv2.imshow('uv_cr', img_uv_cr)

        img_score_map = cvt_float_img(score_maps[:, :, 0])
        cv2.imshow('score_map', img_score_map)

        xyz_cano, rot_inv, xyz_rel = estimate_pose(score_maps, is_right_hand)

        axes.cla()
        plot_hand_3d(axes, xyz_rel)
        img_xyz_rel = get_fig_img(fig)[:, :, ::-1].copy()
        cv2.imshow('xyz_rel', img_xyz_rel)

        plt_showed = False
        saved = False
        while True:
            key = cv2.waitKey(0)
            if chr(key) == 'p':
                if not plt_showed:
                    plt.show()
                    fig = plt.figure()
                    axes = fig.add_subplot(111, projection='3d')
                    plt_showed = True

            if chr(key) == 's':
                if not saved:
                    _save(img_boxes)
                    _save(img_cr)
                    _save(img_uv_cr)
                    _save(img_score_map)
                    _save(img_xyz_rel)
                    print('\tSaved')
                    saved = True

            if key in [ord(i) for i in 'nq']:
                break

        if chr(key) == 'q':
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    _1 = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'y.bmp']
    _2 = [True, True, True, True, True, True, True, True]
    run(_1, _2)
