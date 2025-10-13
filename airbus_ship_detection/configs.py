import torch
from pathlib import Path
import os


# Directories
DIR_DATA = Path("~/data/airbus-ship-detection").expanduser()
DIR_DATA_TRAIN_IMG = DIR_DATA / "train_v2"
DIR_DATA_TEST_IMG = DIR_DATA / "test_v2"
DIR_MODELS = Path("./models")
DIR_LOGS = Path("./logs")

# Create directories if they do not exist
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_LOGS, exist_ok=True)

# Processing parameters
RATIO_IMG_WO_SHIPS = 0.05  # ratio of images without ships to keep

# Training parameters
# Booleans
SHOW_PIXELS_DIST = False
SHOW_SHIP_DIAG = False
SHOW_IMG_LOADER = False

# Training variables
MODEL = 'UNET'  # UNET | IUNET | UNET_RESNET34ImgNet
BATCH_SZ_TRAIN = 16
BATCH_SZ_VALID = 4
LR = 1e-4
N_EPOCHS = 10
TRAIN_SIZE = 1000
TEST_SIZE = 200
RANDOM_STATE = 42
RUN_ID = 1000 # identifier for the training run, used for model and log file names


# Define loss function
LOSS = 'BCEWithDigits' # BCEWithDigits | FocalLossWithDigits | BCEDiceWithLogitsLoss | BCEJaccardWithLogitsLoss

# Define model
MODEL_SEG = 'UNET_RESNET34ImgNet' # UNET | IUNET | UNET_RESNET34ImgNet 
FREEZE_RESNET = False   # if UNET_RESNET34ImgNet

