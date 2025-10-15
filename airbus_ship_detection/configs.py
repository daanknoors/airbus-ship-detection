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
RATIO_IMG_WO_SHIPS = 0.01  # ratio of images without ships to keep
IMG_H = 768
IMG_W = 768

# Training parameters
# Booleans
SHOW_PIXELS_DIST = False
SHOW_SHIP_DIAG = False
SHOW_IMG_LOADER = False

# Training variables
MODEL = 'UNET_CUSTOM'  # UNET | IUNET | UNET_RESNET34ImgNet | UNET_CUSTOM
BATCH_SZ_TRAIN = 16
BATCH_SZ_VALID = 4
LR = 1e-4
N_EPOCHS = 5
TRAIN_SIZE = 2500
TEST_SIZE = 250
RANDOM_STATE = 42
RUN_ID = 5 # identifier for the training run, used for model and log file names


# Define loss function
LOSS = 'BCEWithDigits' # BCEWithDigits | FocalLossWithDigits | BCEDiceWithLogitsLoss | BCEJaccardWithLogitsLoss


