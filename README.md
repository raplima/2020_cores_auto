# Instance segmentation for core picture intepretation

This is a work in progress. We use instance segmentation models as an aid to core interpretation. Instance segmentation models label the pixels in an image (segmentation) and identify what to what object the pixels belong (instance). For the lithofacies interpretation, geologists describe the rock and define what sections belong to what lithofacies:
![alt text](./example-interpretation.jpg "Example of lithofacies interpretation")
The instance segmentation model then can use the core photograph and the polygons defined by the geologist to create a mapping from core photographs to lithofacies interpretation:
[Example of lithofacies interpretation - geologist vs trained model](./example-prediction.jpg)

## Summary

  - [Getting Started](#getting-started)
  - [Contributing](#contributing)
  - [Known issues](#known-issues)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

The [experiment.ipynb](scripts/experiment.ipynb) notebook shows how lithofacies photographs can be used as input to [Mask R-CNN](https://arxiv.org/abs/1703.06870)
to acelerate interpretation. The notebook can be executed in [Google Colab](https://colab.research.google.com). All of the code for analysis is located at the [scripts](scripts) folder. 

The experiments are conducted with [Detectron2](https://github.com/facebookresearch/detectron2) and executed in [Google Colab](https://colab.research.google.com).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Known Issues
- Scripts should be divided into `train.py` and `eval.py`.
- Ideally `config` files should be used (e.g., [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose/configs)).
- Evaluation works right after training, results seem inconsistent when the model is loaded from disk.

## Authors

  - **Rafael Pires de Lima**
  - **Fnu Suriamin**

See also the list of
[contributors](https://github.com/raplima/2020_cores_object_detection/graphs/contributors)
who participated in this project.

## License

This project is licensed under the [BSD 3-Clause](LICENSE.md)
 - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments
  - Most of the scripts are modifications from [Detectron2](https://github.com/facebookresearch/detectron2) [Colab](https://github.com/facebookresearch/detectron2) and other online examples (e.g., [here](https://towardsdatascience.com/how-to-train-detectron2-on-custom-object-detection-data-be9d1c233e4), and [here](https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb#scrollTo=tVJoOm6LVJwW).
  - README was modified from *Billie Thompson's* ([PurpleBooth](https://github.com/PurpleBooth)) template.     
