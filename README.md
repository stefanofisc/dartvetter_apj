# DART-Vetter: A Convolutional Neural Network-based model for Exoplanets Detection
## Overview
This repository contains the code for implementing DART-Vetter, a Convolutional Neural Network (CNN) designed to classify Threshold Crossing Events (TCEs) as either planetary transits or false positives. The model is tailored for photometric data collected from space-based missions like NASA's Kepler and Transiting Exoplanet Survey Satellite (TESS).

## Features
- **Binary/Multi-class classification**: in binary classification mode, the model distinguishes planetary transits and contact eclipsing binaries from other phenomena like stellar variability, instrumental artifacts and non-contact eclipsing binaries. With a simple change in the configuration file and using a suitable dataset, the model is able to solve multi-class classification tasks.
- **Configurable architecture**: the network architecture is highly customizable, with parameters defined in an external configuration file.
- **Ease of use**: includes scripts for training, testing, and evaluating the model on custom datasets.
- **Visualization tools**: generates performance metrics like PR curves and ROC curves.

## Usage
- **Configuration**: the model parameters and dataset paths are defined in the `config_cnn.yaml` file. Update the file with your dataset paths and desired hyperparameters;
- **Input data**: you can download the input global views from the link provided in `link_to_input_data.txt`. Given a csv file, each row has this structure: (**x**,y). **x** is the 201-dimensional input vector and y is its relative label;
- **Training**: use `train_test_cnn_class.py` to train and test the model;
- **Model architecture**: the CNN architecture is defined in m1_class.py;
- **Output**: Model checkpoints, logs and plots will be saved to the directory specified in the configuration file.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/username/dart-vetter.git
cd dart-vetter
```
2. Install required Python packages:
```bash
pip install -r requirements.txt
```

If case of any error during step 2, try to run this command:
```bash
conda env create -f dartvetter_apj.yml
```
