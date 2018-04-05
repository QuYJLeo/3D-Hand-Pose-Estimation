import os
import re
import glob
import numpy as np
import scipy.io as sio
import PIL
import pickle

from utils.ProgressMsgDisplayer import ProgressMsgDisplayer


class Example:
    """Store `key` : `value` pairs.

    By convention, keys should be:
        'img_file', 'boxes', 'uv', 'xyz', 'vis', 'is_right_hand', 'K'
    """

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, ss):
        if type(ss) == str:
            return self._data.get(ss)
        else:
            return [self._data.get(s) for s in ss]


class DatasetReader(object):
    def _example_generator_fn(self):
        raise NotImplementedError()

    def __iter__(self):
        return self._example_generator_fn()


class EReader(DatasetReader):
    def __init__(self, data_dir, verbose=False):
        self.data_dir = data_dir
        self.verbose = verbose

        assert os.path.exists(os.path.join(self.data_dir, '_LABELLED_SAMPLES'))

    def _example_generator_fn(self):
        img_dirs = glob.glob(os.path.join(self.data_dir, '_LABELLED_SAMPLES', '*', ''))

        for img_dir in img_dirs:
            if self.verbose:
                print(img_dir + ':')

            with ProgressMsgDisplayer(not self.verbose) as progress_msg_displayer:
                img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

                anno_file = os.path.join(img_dir, 'polygons.mat')
                anno = sio.loadmat(anno_file)
                polygons = anno['polygons'][0]

                assert len(img_files) == polygons.shape[0]

                for idx, (img_file, masks) in enumerate(zip(img_files, polygons)):
                    boxes = []

                    for mask in masks:
                        if mask.size == 0:
                            continue
                        x_min = np.min(mask[:, 0]) - 1
                        x_max = np.max(mask[:, 0]) - 1
                        y_min = np.min(mask[:, 1]) - 1
                        y_max = np.max(mask[:, 1]) - 1

                        boxes.append({
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max
                        })

                    progress_msg_displayer.update('%06d/%06d %s' %
                                                  (idx + 1,
                                                   len(img_files),
                                                   os.path.basename(img_file)))

                    yield Example(img_file=img_file, boxes=boxes)


class RReader(DatasetReader):
    def __init__(self, data_dir, verbose=False, WEIGHT_NUM_KEY_POINTS=250):
        self.data_dir = data_dir
        self.verbose = verbose

        assert os.path.exists(os.path.join(self.data_dir, 'evaluation'))

        # Used in figuring out the dominant hand of a R example
        self.WEIGHT_NUM_KEY_POINTS = WEIGHT_NUM_KEY_POINTS

    def _get_bounding_boxes(self, img_mask):
        img_mask_left = np.logical_and(2 <= img_mask, img_mask <= 17)
        img_mask_right = np.logical_and(18 <= img_mask, img_mask <= 33)

        boxes = []
        for mask in [img_mask_left, img_mask_right]:
            if np.any(mask):
                y_indices, x_indices = np.where(mask)

                x_min = np.min(x_indices)
                x_max = np.max(x_indices)
                y_min = np.min(y_indices)
                y_max = np.max(y_indices)

                boxes.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max
                })
            else:
                boxes.append(None)
        return boxes

    def _get_pt_vis(self, img_shape, pt):
        img_height, img_width = img_shape

        return 1.0 if (0 <= pt[0] < img_width) and (0 <= pt[1] < img_height) else 0.0

    def _get_domi_hand(self, uv, xyz, vis):
        # figure out the dominant hand (i.e. the hand which has bigger bounding box)
        vis_l = vis[0:21]
        vis_r = vis[21:42]

        uv_l = uv[0:21, :]
        uv_r = uv[21:42, :]

        uv_l = uv_l[np.where(np.greater(vis_l, 0.5))]
        uv_r = uv_r[np.where(np.greater(vis_r, 0.5))]

        assert uv_l.shape[0] > 0 or uv_r.shape[0] > 0

        if uv_l.shape[0] == 0:
            is_right_hand = True
        elif uv_r.shape[0] == 0:
            is_right_hand = False
        else:
            box_lt_l_x, box_lt_l_y = np.min(uv_l, axis=0)
            box_rb_l_x, box_rb_l_y = np.max(uv_l, axis=0)

            box_lt_r_x, box_lt_r_y = np.min(uv_r, axis=0)
            box_rb_r_x, box_rb_r_y = np.max(uv_r, axis=0)

            box_width_l = box_rb_l_x - box_lt_l_x
            box_height_l = box_rb_l_y - box_lt_l_y
            box_area_l = box_width_l * box_height_l

            box_width_r = box_rb_r_x - box_lt_r_x
            box_height_r = box_rb_r_y - box_lt_r_y
            box_area_r = box_width_r * box_height_r

            score_l = box_area_l + self.WEIGHT_NUM_KEY_POINTS * uv_l.shape[0]
            score_r = box_area_r + self.WEIGHT_NUM_KEY_POINTS * uv_r.shape[0]

            is_right_hand = score_r > score_l

        if is_right_hand:
            uv = uv[21:42, :]
            xyz = xyz[21:42, :]
            vis = vis_r
        else:
            uv = uv[0:21, :]
            xyz = xyz[0:21, :]
            vis = vis_l

        return uv, xyz, vis, is_right_hand

    def _example_generator_fn(self):
        img_dirs = glob.glob(os.path.join(self.data_dir, '*', ''))

        for img_dir in img_dirs:
            if self.verbose:
                print(img_dir + ':')

            with ProgressMsgDisplayer(not self.verbose) as progress_msg_displayer:
                img_color_files = sorted(glob.glob(os.path.join(img_dir, 'color', '*.png')))
                img_mask_files = sorted(glob.glob(os.path.join(img_dir, 'mask', '*.png')))

                assert len(img_color_files) == len(img_mask_files)

                anno_file = glob.glob(os.path.join(img_dir, 'anno_*.pickle'))[0]
                with open(anno_file, 'rb') as f:
                    anno = pickle.load(f)

                assert len(img_color_files) == len(anno)

                for idx in range(len(anno)):
                    img_color_file = img_color_files[idx]
                    img_mask_file = img_mask_files[idx]

                    img_mask = np.asarray(PIL.Image.open(img_mask_file))
                    boxes = self._get_bounding_boxes(img_mask)

                    uv_vis, xyz, K = anno[idx]['uv_vis'], anno[idx]['xyz'], anno[idx]['K']
                    uv = uv_vis[:, :2]
                    vis = uv_vis[:, 2]

                    # handle coordinates of palm
                    uv[0, :] = 0.5 * (uv[0, :] + uv[12, :])
                    xyz[0, :] = 0.5 * (xyz[0, :] + xyz[12, :])
                    vis[0] = self._get_pt_vis(img_mask.shape, uv[0, :])

                    uv[21, :] = 0.5 * (uv[21, :] + uv[33, :])
                    xyz[21, :] = 0.5 * (xyz[21, :] + xyz[33, :])
                    vis[21] = self._get_pt_vis(img_mask.shape, uv[21, :])

                    uv, xyz, vis, is_right_hand = self._get_domi_hand(uv, xyz, vis)

                    progress_msg_displayer.update('%06d/%06d %s' %
                                                  (idx + 1,
                                                   len(img_color_files),
                                                   os.path.basename(img_color_file)))

                    yield Example(img_file=img_color_file, boxes=boxes, uv=uv,
                                  xyz=xyz, vis=vis, is_right_hand=is_right_hand, K=K)


class SReader(DatasetReader):
    def __init__(self, data_dir, verbose=False):
        self.data_dir = data_dir
        self.verbose = verbose

        assert os.path.exists(os.path.join(self.data_dir, 'images'))

    def _get_key_points(self, this_anno):
        fx = 822.79041
        fy = 822.79041
        tx = 318.47345
        ty = 250.31296
        baseline = 120.054

        R_r = np.array([
            [1, 0, 0, -baseline],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)

        K = np.array([
            [fx, 0, tx],
            [0, fy, ty],
            [0,  0,  1]
        ])

        xyz_left = this_anno
        xyz_left_homo = np.concatenate([xyz_left, np.ones([1, 21])])

        uv_left_homo = K.dot(xyz_left)

        xyz_right = R_r.dot(xyz_left_homo)

        uv_right_homo = K.dot(xyz_right)

        uv_left = uv_left_homo[:2, :] / uv_left_homo[2:, :]
        uv_right = uv_right_homo[:2, :] / uv_right_homo[2:, :]

        return uv_left.T, uv_right.T, xyz_left.T, xyz_right.T, K

    def _example_generator_fn(self):
        img_dirs = glob.glob(os.path.join(self.data_dir, 'images', '*', ''))

        pattern_scene = re.compile(r'^.*images[/\\](.*)[/\\]$')
        scenes = list(map(lambda x: pattern_scene.match(x).group(1), img_dirs))

        pattern_img_files = re.compile(r'.*_([0-9]+)\.png$')

        def sort_func_img_files(s):
            res = pattern_img_files.search(s)
            return int(res.group(1))

        # dataset S only contains images of left hand
        is_right_hand = False
        # and all 21 keypoints are visible
        vis = np.ones([21], dtype=np.float32)

        for scene in scenes:
            img_dir = os.path.join(self.data_dir, 'images', scene, '')

            if self.verbose:
                print(img_dir + ':')

            with ProgressMsgDisplayer(not self.verbose) as progress_msg_displayer:
                img_left_files = sorted(glob.glob(os.path.join(img_dir, 'BB_left_*.png')),
                                        key=sort_func_img_files)
                img_right_files = sorted(glob.glob(os.path.join(img_dir, 'BB_right_*.png')),
                                         key=sort_func_img_files)

                assert len(img_left_files) == len(img_right_files)

                anno_file = os.path.join(self.data_dir, 'labels', '%s_BB.mat' % scene)
                anno = sio.loadmat(anno_file)['handPara']

                assert len(img_left_files) == anno.shape[2]

                for idx in range(len(img_left_files)):
                    img_left_file, img_right_file = img_left_files[idx], img_right_files[idx]

                    uv_left, uv_right, xyz_left, xyz_right, K = self._get_key_points(anno[:, :, idx])

                    # to match dataset R's convention
                    tmp = [0] + list(range(20, 0, -1))
                    uv_left = uv_left[tmp, :]
                    xyz_left = xyz_left[tmp, :]
                    uv_right = uv_right[tmp, :]
                    xyz_right = xyz_right[tmp, :]

                    # `xyz`s are in millimeters in dataset S, but in meters in dataset R
                    xyz_left /= 1000.0
                    xyz_right /= 1000.0

                    progress_msg_displayer.update('%06d/%06d %s %s' % (
                                idx + 1,
                                len(img_left_files),
                                os.path.basename(img_left_file),
                                os.path.basename(img_right_file)))

                    to_yield = []
                    to_yield.append(Example(img_file=img_left_file,
                                            uv=uv_left, xyz=xyz_left, K=K,
                                            is_right_hand=is_right_hand, vis=vis))
                    to_yield.append(Example(img_file=img_right_file,
                                            uv=uv_right, xyz=xyz_right, K=K,
                                            is_right_hand=is_right_hand, vis=vis))
                    yield to_yield


def get_examples(which_dataset):
    assert which_dataset in ['E', 'R', 'S']

    pickle_path = os.path.join('..', 'data', '%s.pickle' % which_dataset)
    data_dir = os.path.join('..', 'data', which_dataset)

    TheDatasetReader = {'E': EReader, 'R': RReader, 'S': SReader}[which_dataset]

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as ifs:
            examples = pickle.load(ifs)
    else:
        print('')
        print('Reading %s examples ......' % which_dataset)
        print('-------------------------')
        examples = list(TheDatasetReader(data_dir=data_dir, verbose=True))

        with open(pickle_path, 'wb') as ofs:
            pickle.dump(examples, ofs)

    return examples


def _test_ereader():
    cv2.namedWindow('boxes')

    for example in EReader(data_dir='../data/E', verbose=True):
        img_file, boxes = example['img_file', 'boxes']
        img = cv2.imread(img_file)

        for box in boxes:
            x_min, x_max, y_min, y_max = [
                int(box[x]) for x in ('x_min', 'x_max', 'y_min', 'y_max')
            ]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imshow('boxes', img)

        while True:
            key = cv2.waitKey(0)
            if key in [ord(i) for i in 'nq']:
                break

        if chr(key) == 'q':
            break

    cv2.destroyAllWindows()


def _test_rreader():
    cv2.namedWindow('boxes')
    cv2.namedWindow('uv')

    for example in RReader(data_dir='../data/R', verbose=True):
        img_file, boxes, uv, xyz, vis, is_right_hand, K = example[
            'img_file', 'boxes', 'uv', 'xyz', 'vis', 'is_right_hand', 'K'
        ]
        img = cv2.imread(img_file)

        img_boxes = np.copy(img)
        for box in boxes:
            if box is None:
                continue
            x_min, x_max, y_min, y_max = [
                int(box[x]) for x in (
                    'x_min', 'x_max', 'y_min', 'y_max'
                )
            ]
            cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imshow('boxes', img_boxes)

        u1v1_homo = np.dot(K, xyz.T).T
        u1v1 = u1v1_homo[:, :2] / u1v1_homo[:, 2:]

        img_uv = np.copy(img)
        for this_uv, this_u1v1, this_vis in zip(uv, u1v1, vis):
            if this_vis == 0:
                continue

            this_uv = this_uv.astype(np.int64)
            this_u1v1 = this_uv.astype(np.int64)

            cv2.rectangle(img_uv,
                          (this_uv[0] - 1, this_uv[1] - 1),
                          (this_uv[0] + 1, this_uv[1] + 1),
                          (0, 255, 0), 1)
            cv2.rectangle(img_uv,
                          (this_u1v1[0] - 1, this_u1v1[1] - 1),
                          (this_u1v1[0] + 1, this_u1v1[1] + 1),
                          (255, 255, 255), 1)

        cv2.putText(img_uv,
                    'R' if is_right_hand else 'L',
                    (0, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('uv', img_uv)

        while True:
            key = cv2.waitKey(0)
            if key in [ord(i) for i in 'nq']:
                break

        if chr(key) == 'q':
            break

    cv2.destroyAllWindows()


def _test_sreader():
    cv2.namedWindow('uv_left')
    cv2.namedWindow('uv_right')

    for example_left, example_right in SReader(data_dir='../data/S', verbose=True):
        img_left_file, uv_left, xyz_left, K = example_left['img_file', 'uv', 'xyz', 'K']
        img_right_file, uv_right, xyz_right, _ = example_right['img_file', 'uv', 'xyz', 'K']

        img_left = cv2.imread(img_left_file)
        img_right = cv2.imread(img_right_file)

        for img, uv, xyz in zip((img_left, img_right), (uv_left, uv_right), (xyz_left, xyz_right)):
            u1v1_homo = np.dot(K, xyz.T).T
            u1v1 = u1v1_homo[:, :2] / u1v1_homo[:, 2:]

            for this_uv, this_u1v1 in zip(uv, u1v1):
                this_uv = this_uv.astype(np.int64)
                this_u1v1 = this_uv.astype(np.int64)

                cv2.rectangle(img,
                              (this_uv[0] - 1, this_uv[1] - 1),
                              (this_uv[0] + 1, this_uv[1] + 1),
                              (0, 255, 0), 1)
                cv2.rectangle(img,
                              (this_u1v1[0] - 1, this_u1v1[1] - 1),
                              (this_u1v1[0] + 1, this_u1v1[1] + 1),
                              (255, 255, 255), 1)

        cv2.imshow('uv_left', img_left)
        cv2.imshow('uv_right', img_right)

        while True:
            key = cv2.waitKey(0)
            if key in [ord(i) for i in 'nq']:
                break

        if chr(key) == 'q':
            break

    cv2.destroyAllWindows()


def _test_progress_msg_display():
    print('')
    print('Reading E examples ......')
    list(EReader(data_dir='../data/E', verbose=True))

    print('')
    print('Reading R examples ......')
    list(RReader(data_dir='../data/R', verbose=True))

    print('')
    print('Reading S examples ......')
    list(SReader(data_dir='../data/S', verbose=True))


if __name__ == '__main__':
    import cv2

    _test_sreader()
