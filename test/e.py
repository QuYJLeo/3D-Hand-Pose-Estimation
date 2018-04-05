import os

import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

from utils.DatasetReader import EReader

from test.run import get_save_func, detect_hand
from test.config import config


DIR = '../__backup/test/e'


def main():
    cv2.namedWindow('boxes')

    for example in EReader(data_dir='../data/E', verbose=True):
        img_file, boxes = example['img_file', 'boxes']
        img = np.array(PIL.Image.open(img_file))
        img_H, img_W, _ = img.shape

        _save = get_save_func(os.path.join(DIR, os.path.basename(img_file)))

        img_boxes = img[:, :, ::-1].copy()
        for box in boxes:
            if box is None:
                continue
            x_min, x_max, y_min, y_max = [
                int(box[x]) for x in ('x_min', 'x_max', 'y_min', 'y_max')
            ]
            cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        cv2.imshow('boxes', img_boxes)

        while True:
            key = cv2.waitKey(0)
            if key in [ord(i) for i in 'nsq']:
                break

        if chr(key) == 'q':
            break
        if chr(key) == 's':
            _save(img_boxes)

            boxes, scores = detect_hand(img)
            boxes, scores = zip(*filter(lambda _: _[1] >= config['hand_detection']['th1'], zip(boxes, scores)))

            img_detected_boxes = img[:, :, ::-1].copy()
            for box, score in zip(boxes, scores):
                pt1 = (int(box[1] * img_W), int(box[0] * img_H))
                pt2 = (int(box[3] * img_W), int(box[2] * img_H))
                cv2.rectangle(img_detected_boxes, pt1, pt2, (255, 255, 255), 2)

                pt = (int(box[1] * img_W), int(box[2] * img_H - 2))
                cv2.putText(img_detected_boxes, '%d%%' % int(score * 100), pt, cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
            _save(img_detected_boxes)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
