# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 07:35:25 2020

@author: Rafael
"""
import os
from shutil import copy2
import json
import argparse

from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg

# to train
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


from data_loading import get_data_dicts


def main(data_dir, dataset_tag, fold_idx, dset):
    setup_logger()

    # read the classes dictionary
    json_file = os.path.join(data_dir, "classes.json")
    with open(json_file) as f:
        classes = json.load(f)
    
    print(f'setting fold {fold_idx}')
    for d in ["train", "val"]:
        tag = f'{dataset_tag}_fold_{fold_idx}_'
        # don't use fully f strings for register, it merges 'train' and 'val'
        print(f'\t {tag}' + d)
        DatasetCatalog.register(tag+d, lambda d=d: get_data_dicts(data_dir, tag + d + '.json'))
        MetadataCatalog.get(tag+d).set(thing_classes=sorted([it for _, it in classes.items()]))

    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{dataset_tag}_train_{fold_idx}_fold",)
    cfg.DATASETS.TEST = (f"{dataset_tag}_val_{fold_idx}_fold",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
      
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)   # number of classes
    
    cfg.OUTPUT_DIR = f'output_{fold_idx}'
    
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_fold_{fold_idx}.pth")

    cfg.DATASETS.TEST = (f"{dataset_tag}_{dset}_{fold_idx}_fold",)
    evaluator = COCOEvaluator(f"{dataset_tag}_{dset}_{fold_idx}_fold", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, f"{dataset_tag}_{dset}_{fold_idx}_fold")
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
    with open(os.path.join(cfg.OUTPUT_DIR, f"cocoeval_{dset}_{fold_idx}.json"), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-d", "--data_dir", required=True, help="data directory")
    ap.add_argument("-dt", "--dataset_tag", required=True, help="task tag")
    ap.add_argument("-i", "--fold_idx", required=True, help="fold index")
    ap.add_argument("-dset", "--dset", required=True, help="dataset tag (train, val)")    
    args = ap.parse_args()
    main(args.data_dir, args.dataset_tag, args.fold_idx, args.dset)