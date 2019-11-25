# -*- coding: utf-8 -*-
# @__ramraj__

# WBCRecognition without Dask & allow-soft-GPU-placement


import os
import cv2
import sys
import json
import time
import logging
import itertools
import numpy as np
import pandas as pd


import large_image
import utils as cli_utils
from ctk_cli import CLIArgumentParser
import histomicstk.utils as htk_utils

sys.path.append('..')


from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork


from classify import classify_main
from classify import classify_cellscropping


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logging.basicConfig(level=logging.CRITICAL)

sys.path.append(os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')))


CONFIG = '../luminoth/examples/sample_config.yml'
PATH = '../91316_leica_at2_40x.svs.38552.50251.624.488.jpg'
CKPT = '../classify/ckpt/'
MAX_DET = 1000
MIN_PROB = 0.1


def detect_tile_cell(slide_path, tile_position, csv_dict, args, it_kwargs):

    start_t = time.time()

    print('--- Loading Image...')
    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    # get tile image
    im_tile = tile_info['tile'][:, :, :3]
    t1 = time.time() - start_t
    csv_dict['Image Loading'].append(round(t1, 3))
    print('--- Finished Loading Image')

    cv2.imwrite('hey_im_tile.png', im_tile)

    # *******************************************************
    #
    # Perform cell detections
    #
    # ########################### DETECTION #################
    print('--- Performing cell detections...')

    config = get_config(CONFIG)
    if not args.max_det is None:
        config.model.rcnn.proposals.total_max_detections = args.max_det
    else:
        config.model.rcnn.proposals.total_max_detections = MAX_DET
    if not args.min_prob is None:
        config.model.rcnn.proposals.min_prob_threshold = args.min_prob
    else:
        config.model.rcnn.proposals.min_prob_threshold = MIN_PROB

    print('--- Currently Analysing Input Image Size : ', im_tile.shape)

    network = PredictorNetwork(config)
    objects = network.predict_image(im_tile)
    print('--- Finished Cell Detections')
    t2 = time.time() - start_t
    t22 = float(t2) - float(t1)
    csv_dict['Cell Detection'].append(round(t22, 3))

    print('***** Number of Detected Cells ****** : ', len(objects))

    #
    # Perform JSON loading
    #
    print('--- Performing Cell Crops loading...')
    im_tile_rgb = cv2.cvtColor(im_tile, cv2.COLOR_BGR2RGB)
    if not args.inputImageFile is None:
        _, images, ids = classify_cellscropping.\
            load_json(im_tile_rgb, preds=objects)
    else:
        _, images, ids = classify_cellscropping.\
            load_json(im_tile_rgb, preds=objects)
    print('--- Finished Cell Crops loading')
    t3 = time.time() - start_t
    t33 = float(t3) - float(t2)
    csv_dict['Cell Cropping'].append(round(t33, 3))
    csv_dict['Number of Cells'].append(len(ids))

    #       ########################### CLASSIFICATION #######################
    print('--- Performing Cell Classification...')
    try:
        final_json = classify_main.predict(images, ids, CKPT)
    except ValueError:
        final_json = []
        print('!!!!! Can not Conduct Classification on 0 Number of Cells Detected !!!!!')
    print('--- Finished Cell Classification')
    t4 = time.time() - start_t
    t44 = float(t4) - float(t3)
    csv_dict['Cell Classification'].append(round(t44, 3))

    # # Delete border nuclei
    # if args.ignore_border_nuclei is True:
    #     im_nuclei_seg_mask = htk_seg_label.delete_border(im_nuclei_seg_mask)

    # generate cell annotations
    cell_annot_list = cli_utils.create_tile_cell_annotations(
        final_json, tile_info, args.cell_annotation_format)
    t5 = time.time() - start_t
    t55 = float(t5) - float(t4)
    csv_dict['Annotation Writing'].append(round(t55, 3))

    return cell_annot_list, csv_dict


def main(args):

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise IOError('Input image file does not exist.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    start_time = time.time()

    #
    # Read Input Image
    #
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)

    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))

    is_wsi = ts_metadata['magnification'] is not None

    #
    # Compute tissue/foreground mask at low-res for whole slide images
    #
    if is_wsi and process_whole_image:

        print('\n>> Computing tissue/foreground mask at low-res ...\n')

        start_time = time.time()

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        fgnd_time = time.time() - start_time

        print('low-res foreground mask computation time = {}'.format(
            cli_utils.disp_time_hms(fgnd_time)))

    #
    # Compute foreground fraction of tiles in parallel
    #
    tile_fgnd_frac_list = [1.0]

    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    if not process_whole_image:

        it_kwargs['region'] = {
            'left':   args.analysis_roi[0],
            'top':    args.analysis_roi[1],
            'width':  args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units':  'base_pixels'
        }

    if is_wsi:

        print('\n>> Computing foreground fraction of all tiles ...\n')

        start_time = time.time()

        num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

        print('Number of tiles = {}'.format(num_tiles))

        if process_whole_image:

            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
                it_kwargs
            )

        else:

            tile_fgnd_frac_list = np.full(num_tiles, 1.0)

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        fgnd_frac_comp_time = time.time() - start_time

        print('Number of foreground tiles = {0:d} ({1:2f}%%)'.format(
            num_fgnd_tiles, percent_fgnd_tiles))

        print('Tile foreground fraction computation time = {}'.format(
            cli_utils.disp_time_hms(fgnd_frac_comp_time)))

    #
    # Detect cell in parallel
    #
    print('\n>> Detecting cell ...\n')

    start_time = time.time()

    tile_cell_list = []

    csv_dict = {}
    csv_dict['Image Loading'] = []
    csv_dict['Cell Detection'] = []
    csv_dict['Cell Cropping'] = []
    csv_dict['Cell Classification'] = []
    csv_dict['Annotation Writing'] = []
    csv_dict['Number of Cells'] = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        cur_cell_list, csv_dict = detect_tile_cell(
            args.inputImageFile,
            tile_position,
            csv_dict,
            args, it_kwargs)
        # append result to list
        tile_cell_list.append(cur_cell_list)

        df = pd.DataFrame(csv_dict, columns=['Number of Cells',
                                             'Image Loading',
                                             'Cell Detection',
                                             'Cell Cropping',
                                             'Cell Classification',
                                             'Annotation Writing'])
        df.to_csv('%s.csv' % args.outputCellAnnotationFile[:-5])

    cell_list = list(itertools.chain.from_iterable(tile_cell_list))

    cell_detection_time = time.time() - start_time

    print('Number of cells = {}'.format(len(cell_list)))

    print('Cell detection time = {}'.format(
        cli_utils.disp_time_hms(cell_detection_time)))

    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputCellAnnotationFile))[0]

    annotation = {
        "name":     annot_fname + '-cell-' + args.cell_annotation_format,
        "elements": cell_list
    }

    with open(args.outputCellAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":

    main(CLIArgumentParser().parse_args())
