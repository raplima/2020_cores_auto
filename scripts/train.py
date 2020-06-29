# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 07:35:25 2020

@author: Rafael
"""


# setup evaluator for the trainer
class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_dir="./output/"):
    if output_dir is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_dir)

# print GPU information
!nvidia-smi
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (f"{dataset_tag}_train_{fold_idx}_fold",)
cfg.DATASETS.TEST = (f"{dataset_tag}_train_{fold_idx}_fold", f"{dataset_tag}_val_{fold_idx}_fold",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.INPUT.MAX_SIZE_TRAIN = 1000

# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001  
cfg.SOLVER.MAX_ITER = 200    
cfg.TEST.EVAL_PERIOD = 100

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)   # number of classes

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=False)

# train:
trainer.train()

# copy the weights 
copy2(os.path.join(cfg.OUTPUT_DIR, 'model_final.pth'), 
        os.path.join(cfg.OUTPUT_DIR, f'model_fold_{fold_idx}.pth'))  