Many of the scripts are modifications from [Detectron2](https://github.com/facebookresearch/detectron2) [Colab](https://github.com/facebookresearch/detectron2) and other online examples (e.g., [here](https://towardsdatascience.com/how-to-train-detectron2-on-custom-object-detection-data-be9d1c233e4), and [here](https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb#scrollTo=tVJoOm6LVJwW).*** 

Experiments can be conducted using the [`experiment notebook`](./experiment.ipynb). 
The polygon format we use does not follow COCO structure, thus [`data_loading.py`](/.data_loading.py) contains a function to load the data.
Before the experments we divided the data into five-fold cross-validation using [`data_prep.py`](./data_prep.py)
The experiments were executed in [Google Colab](https://colab.research.google.com) and the execution logs for both ResNet50-C4 and ResNet50-FPN are in two different notebooks: [`executer_R50_C4.ipynb`](./executer_R50_C4.ipynb) and [`executer_R50_FPN.ipynb`](./executer_R50_FPN.ipynb)
