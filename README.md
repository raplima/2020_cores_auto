# Instance segmentation for core picture intepretation

In progress. 

## Summary

  - [Getting Started](#getting-started)
  - [Runing the tests](#running-the-tests)
  - [Deployment](#deployment)
  - [Contributing](#contributing)
  - [Known issues](#known-issues)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

    Give examples

### Installing

A step by step series of examples that tell you how to get a development
env running

Say what the step will be

    Give the example

And repeat

    until finished

End with an example of getting some data out of the system or using it
for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

    Give an example

### And coding style tests

Explain what these tests test and why

    Give an example

## Deployment

Add additional notes about how to deploy this on a live system

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Known Issues
- Scripts should probably be divided into `train.py` and `eval.py`.
- Ideally `config` files should be used (e.g., [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose/configs)).
- Evaluation works right after training, results seem inconsistent when the model is loaded from disk.
## Authors

  - **Rafael Pires de Lima**

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
