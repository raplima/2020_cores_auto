Many of the scripts here are modifications or reproductions from PyTorch github [page](https://github.com/pytorch/).
We pulled their [detection scripts](https://github.com/pytorch/vision/tree/master/references/detection) in May 6, 2020.  
The  [Object detection reference training scripts](#object-detection-reference-training-scripts) section shows their 
[README](https://github.com/pytorch/vision/blob/master/references/detection/README.md) at the time. 
*** 

# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.

### Faster R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

