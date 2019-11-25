# -*- coding: utf-8 -*-
# @__ramraj__


import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import pandas as pd
import os


PIL_or_CV2 = 'PIL'


IMAGE_DIR = '../fulldata/'
PREDICTION_RESULTS = './DISEASED_SAMPLES_RESULTS_WHOLE_96/predictions_agg_classification.csv'
DST_OVERLAY_RESULTS = './DISEASED_SAMPLES_RESULTS_WHOLE_96/OVERLAYS_PIL_POINT/'

if not os.path.exists(DST_OVERLAY_RESULTS):
    os.makedirs(DST_OVERLAY_RESULTS)

COLORS = [(0, 0, 255), (128, 0, 128), (153, 51, 255), (153, 0, 255), (255, 0, 102), (0, 0, 153),
          (102, 0, 255), (255, 80, 80), (0, 102, 204), (102, 102, 153), (0, 204, 255), (0, 0, 0)]

"""
Color Scheme

CLASSES = ['blast', 'band_seg', 'metamyelocyte',
           'myelocyte', 'eosinophil', 'promyelocyte',
           'basophil',
           'erythroid', 'lymphocyte', 'monocyte', 'plasma',
           'unknown']

[(0,0,255) ,(128,0,128), (153,51,255), (153,0,255), (255,0,102), (0,0,153), (102,0,255), (255,80,80), (0,102,204), (102,102,153), (0,204,255),(0,0,0)]

blast = rgb(0,0,255)
band_seg = rgb(128,0,128)
metamyelocyte = rgb(153,51,255)
myelocyte = rgb(153,0,255)
eosinophil = rgb(255,0,102)
promyelocyte = rgb(0,0,153)
basophil = rgb(102,0,255)
erythroid = rgb(255,80,80)
lymphocyte = rgb(0,102,204)
monocyte = rgb(102,102,153)
plasma = rgb(0,204,255)
unknown = rgb(0,0,0)


"""


def vis(image_name):

    if PIL_or_CV2 == 'PIL':
        img = Image.open(os.path.join(IMAGE_DIR, image_name)).convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        img = cv2.imread(os.path.join(IMAGE_DIR, image_name))

    df = pd.read_csv(PREDICTION_RESULTS)

    matching_image_name = '%s.png' % image_name[:-4]
    roi_df = df[df['ImageID'] == matching_image_name]

    image_ids = roi_df['ImageID']
    cell_ids = roi_df['CellID']
    predictions = roi_df['Predictions']

    for cell_index in range(roi_df.shape[0]):
        cell_id = cell_ids.iloc[cell_index]

        tmp_slice = cell_id.split('_')[-2]
        x_cent, y_cent, bbox_h, bbox_w, label = tmp_slice.split('.')
        x_min = int(x_cent) - int(bbox_w) / 2
        x_max = int(x_cent) + int(bbox_w) / 2
        y_min = int(y_cent) - int(bbox_h) / 2
        y_max = int(y_cent) + int(bbox_h) / 2

        prediction = int(predictions.iloc[cell_index])
        line_color = COLORS[prediction]
        # line_color = (0, 255, 0) ## objectness

        if PIL_or_CV2 == 'PIL':
            DIAMETER = 7
            coordinates = [int(x_cent) - DIAMETER, int(y_cent) - DIAMETER,
                           int(x_cent) + DIAMETER, int(y_cent) + DIAMETER]
            draw.ellipse(coordinates,  fill=line_color, outline=line_color)

        else:
            OUTLINE_WIDTH = 3
            coordinates = [x_min, y_min, x_max, y_max]
            for i in range(int(OUTLINE_WIDTH)):
                coords = [
                    coordinates[0] - i,
                    coordinates[1] - i,
                    coordinates[2] + i,
                    coordinates[3] + i,
                ]

                draw.rectangle(coords, outline=line_color)

    dst_img_file = os.path.join(
        DST_OVERLAY_RESULTS, '%s.png' % image_name[:-4])

    if PIL_or_CV2 == 'PIL':
        img.save(dst_img_file, "PNG")
    else:
        cv2.imwrite(dst_img_file, img)


if __name__ == '__main__':

    files = os.listdir(IMAGE_DIR)

    for file in files:
        print file
        vis(image_name=file)
