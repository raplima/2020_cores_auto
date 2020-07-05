# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 07:35:25 2020
@author: Rafael
"""
import os
import json
import argparse
import random
import cv2

import numpy as np
import torch

from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg

# to train
from detectron2.engine import DefaultTrainer

# to evaluate
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# local functions
from data_loading import get_data_dicts
from utils import plot_res

setup_logger()

def main(data_dir, dataset_tag, fold_idx, max_iter):
    # set seed for reproducibility 
    # although it is not guaranteed (https://pytorch.org/docs/stable/notes/randomness.html)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    # read the classes dictionary
    json_file = os.path.join(data_dir, "classes.json")
    with open(json_file) as f:
        classes = json.load(f)
    
    # save a list with same format as function and metadata:
    thing_classes = sorted([it for _, it in classes.items()])
    
    # read the colors dictionary
    json_file = os.path.join(data_dir, "colors_dict.json")
    with open(json_file) as f:
        patch_dict = json.load(f)    
    
    print(f'setting fold {fold_idx}')
    for d in ["train", "val"]:
        tag = f'{dataset_tag}_fold_{fold_idx}_'
        # don't use fully f strings for register, it merges 'train' and 'val'
        print(f'\t {tag}' + d)
        DatasetCatalog.register(tag+d, lambda d=d: get_data_dicts(data_dir, tag + d + '.json'))
        MetadataCatalog.get(tag+d).set(thing_classes=sorted([it for _, it in classes.items()]))
    
    # setup evaluator for the trainer
    class CocoTrainer(DefaultTrainer):
      @classmethod
      def build_evaluator(cls, cfg, dataset_name, output_dir="./coco_train/"):
        if output_dir is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_dir = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_dir)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{dataset_tag}_fold_{fold_idx}_train",)
    cfg.DATASETS.TEST = (f"{dataset_tag}_fold_{fold_idx}_train", f"{dataset_tag}_fold_{fold_idx}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1000

    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.TEST.EVAL_PERIOD = 200

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)   # number of classes

    cfg.OUTPUT_DIR = f'output_fold_{fold_idx}'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    
    # train:
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

    predictor = DefaultPredictor(cfg)

    for d in ['train', 'val']:
        print(f'\n\n**Starting {d} eval**')
        cfg.DATASETS.TEST = (f"{dataset_tag}_fold_{fold_idx}_{d}",)
        evaluator = COCOEvaluator(f"{dataset_tag}_fold_{fold_idx}_{d}", cfg, False, output_dir=f"./coco/")
        val_loader = build_detection_test_loader(cfg, f"{dataset_tag}_fold_{fold_idx}_{d}")
        metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
        with open(os.path.join(cfg.OUTPUT_DIR, f"cocoeval_{d}_{fold_idx}.json"), 'w') as f:
            json.dump(metrics, f)
        
  # save one example:
        samp = random.sample(DatasetCatalog.get(f"{dataset_tag}_fold_{fold_idx}_{d}"), 1)[0]
        print(f"randomly selected {samp['file_name']}")
        im = cv2.imread(samp['file_name'])
        outputs = predictor(im)
        fig = plot_res(cfg, samp, im, outputs, patch_dict, thing_classes)
        fig.tight_layout
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, f"{dataset_tag}_fold_{fold_idx}_{d}_{samp['file_name']}.pdf".replace('/','-')), dpi=1000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-d", "--data_dir", required=True, help="data directory")
    ap.add_argument("-dt", "--dataset_tag", required=True, help="task tag")
    ap.add_argument("-i", "--fold_idx", required=True, help="fold index")
    ap.add_argument("-m", "--max_iter", required=True, help="maximum number of iterations")
    args = ap.parse_args()
    main(args.data_dir, args.dataset_tag, args.fold_idx, args.max_iter)