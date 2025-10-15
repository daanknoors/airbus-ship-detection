# Airbus Ship Detection
This repository contains code and notebooks for the Airbus Ship Detection Challenge on Kaggle. The goal of the challenge is to build a model that can accurately detect ships in satellite images.

Original challenge link: https://www.kaggle.com/competitions/airbus-ship-detection/overview

## Assignment
This project was part of an assignment aimed to test skills in computer vision, deep learning, and model deployment. The task involved building a model to detect ships in satellite images and deploying it as a web application using FastAPI. 

It consists of two main parts: 
- **Part 1: model training**: load satellite images, preprocess data, build and train a deep learning model, evaluate its performance using metrics such as F2 score and Intersection over Union (IoU), and save the trained model.
- **Part 2: model deployment**: create an openapi application that allows users to upload an image and get binary mask of detected ships, as well as a contour displayed on the original image.


## Summary of work done
There was a **6 hour time limit** to complete the assignment. Hence, the code is not completely polished and there are several areas for improvement. 

I also wanted to take this opportunity to experiment with libraries such as Polars, PyTorch and FastAPI. Some parts of the code were re-used from the original kaggle challenge to save time, but still had to be re-implemented to work with these libraries.

The main tasks completed and next steps for improvement are summarized below:

### 🧱 Project Setup
- Organized repo structure and dependencies with Poetry.
- Downloaded data from Kaggle.
- Set up notebooks for EDA and training.

Next steps:
- Improve code modularity, documentation, and logging.
- Add unit tests and enhance error handling.

### 📊 Data Processing
- Implemented data loading and preprocessing.
- Functions for run-length encoding and decoding.
- Simple data augmentations.


Next steps:
- Experiment with different preprocessing techniques.
- Experiment with other class imbalance techniques (Undersampling, Oversampling, SMOTE).
- Add more advanced augmentation.

### 🧩 Model Development
- Built a U-Net model for segmentation.
- Load other pre-trained models (e.g., ResNet34).
- Set up loss functions (BCE, Dice Loss).
- Implemented training and validation loops using PyTorch and GPU acceleration.
- Saved the trained model and training logs.

Next steps:
- Experiment with full dataset, larger batch sizes and more epochs.
- Perform more extensive hyperparameter tuning.
- Save and compare multiple model versions.
- Explore alternative neural network architectures.


### 🖼️ Evaluation and Visualization
- Evaluated using F2 Score and IoU.
- Created functions to visualize images, masks, and predictions.
- Visualized training curves.

Next steps:
- Enhance visualizations with more metrics and comparisons.
- Show mask contours on original images in yellow as per the assignment.

### 🚀 Deployment
- Created a FastAPI app for model inference (partially working).

Next steps:
- Fix and finalize API functionality.
- Deploy to a cloud platform.

## Installation

### Activate environment
Install poetry: https://python-poetry.org/docs/

And the environment:
```
poetry install
poetry shell
```


### Download data
Set up kaggle API credentials to download data, see:
https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md

Run: `kaggle competitions download -c airbus-ship-detection`

## Repo structure
Structure of folders and files in repo
```
airbus-ship-detection/
|-- airbus_ship_detection/ - main package code
|   |-- augmentations.py - data augmentation functions
|   |-- configs.py - configuration file
|   |-- datasets.py - dataset class
|   |-- inference.py - model inference functions
|   |-- losses.py - loss functions
|   |-- main.py - fastapi application
|   |-- metrics.py - evaluation metrics functions
|   |-- models.py - model architectures
|   |-- processing.py - data processing functions
|   |-- train.py - training functions
|   |-- visuals.py - visualization functions
|
|-- models/ - saved models
|
|-- notebooks/ - jupyter notebooks
|   |-- 000_exploratory_data.ipynb - initial data exploration notebook
|   |-- 001_train_model.ipynb - notebook for training and evaluating models
|
|-- tests/ - unit tests
|
|-- README.md - repo documentation
|-- pyproject.toml - poetry configuration file
|-- poetry.lock - poetry lock file
|-- .gitignore - git ignore file
```
