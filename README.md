# Automated core classification with object detection

In progress. 

I initially saw this on
[PurpleBooth](https://github.com/PurpleBooth/a-good-readme-template), 
and I chose to move it here for personal project templates

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

Even with Anaconda, installing pycocotools in Windows might not go smoothly. Here is what we did following [this](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62381):  
1. Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
2. Install cython: ```conda install cython```
3. Install pycocotools:  
    ``` pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI```  

You might need to further edit pycocotools in case you get an numpy error (missing ```int``` conversion)
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
  - A lot of the code for object detection was modified from [PyTorch vision](https://github.com/pytorch/vision/tree/master/references/detection).
  - README was modified from *Billie Thompson's* ([PurpleBooth](https://github.com/PurpleBooth)) template.     
