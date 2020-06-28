# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:22:58 2020

@author: Rafael
"""

import os
import json
import cv2
from detectron2.structures import BoxMode
import numpy as np

def get_data_dicts(img_dir, dict_json):
    """
    function to format the data into detectron2 input parameters

    Parameters
    ----------
    img_dir : string
        path to folder with images and a "classes.json" dictionary.
    dict_json : dict
        a dictionary containing image info.
    Returns
    -------
    dataset_dicts : dict
        the dictionary used as input to detectron2 model
    """
    # get the classes:
    json_file = os.path.join(img_dir, "classes.json")
    with open(json_file) as f:
        ann_classes = json.load(f)
    # transform the dictionary into a list 
    classes=sorted([it for _, it in ann_classes.items()])

    json_file = os.path.join(img_dir, dict_json)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            #assert not anno["region_attributes"]
            region = anno['region_attributes']
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(ann_classes[region['lithofacies']]),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
