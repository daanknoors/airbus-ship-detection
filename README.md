# airbus-ship-detection

## Installation

### Activate environment
Activate poetry environment:
```bash
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
|   |-- configs.py - configuration file
|   |-- processing.py - data processing functions
|   |-- visuals.py - visualization functions
|
|-- notebooks/ - jupyter notebooks
|
|-- tests/ - unit tests
|
|-- models/ - saved models
|
|-- README.md - repo documentation
|-- pyproject.toml - poetry configuration file
|-- poetry.lock - poetry lock file
|-- .gitignore - git ignore file
```