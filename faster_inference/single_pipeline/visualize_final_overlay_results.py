# -*- coding: utf-8 -*-
# @__ramraj__


import cv2
import numpy as np
import json
import os


OUTPUT_DIR = 'Classification_Vis_Outputs'


def vis(json_file, image_file):
    json_data = json.load(open(json_file))
    img = cv2.imread(image_file)

    print len(json_data)

    for obj in json_data:
        x_c, y_c, _ = obj['center']
        h = obj['height']
        w = obj['width']
        linecolor = str(obj['lineColor'])

        line_color = list(map(int, linecolor[4:-1].split(',')))

        x1 = x_c - (h / 2)
        y1 = y_c - (w / 2)
        x2 = x_c + (h / 2)
        y2 = y_c + (w / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), line_color, 2)

    dst_img_file = os.path.join(
        OUTPUT_DIR, 'pred_class_%s' % image_file.split('/')[-1])

    cv2.imwrite(dst_img_file, img)


if __name__ == '__main__':
    jsons_path = 'classification_json_outputs'
    images_path = 'fulldata'
    files = os.listdir(jsons_path)

    for file in files:

        json_path = os.path.join(jsons_path, file)
        image_path = os.path.join(images_path, '%s.jpg' % file[:-5])

        vis(json_path, image_path)
