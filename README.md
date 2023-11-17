<!-- PROJECT LOGO -->
# Traffic Anomaly and Classification
<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to capture traffic and preprocess it in hope we can identify and classify anomalies in traffic for intrusion detection purposes.

## Project Structure

```
├── models (Available Models)/
│   ├── flaml_model/
│   │   ├── flaml_model/
│   │   │   ├── model.py (File containing the main model for package)
│   │   │   ├── utils.py (Utility files)
│   │   │   └── train.py (File to run when training model)
│   │   ├── saved_models/ (Folder containing saved models)
│   │   ├── requirements.txt
│   │   └── setup.py
│   └── ...
├── tools (Data analysis tools and scripts)
├── requirements.txt (Project Dependencies)
├── README.md (Read me file)
├── constants.py (Variables used in build.py for configuration)
├── build.py (Build file to run before running run.py)
└── run.py (File to run to start IDS system)
```

<!-- GETTING STARTED -->
## Prerequisites

### Python
List of prerequisites that should be installed before hand.

1. Have Python installed with pip to create virtual environments.
https://www.python.org/downloads/

2. Create virtual environment with the requirements.txt file at the root level.

### Datasets
List of datasets used:
-   [Intrusion Detection Evaluation Dataset (CIC-IDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)
-   [IPS/IDS dataset on AWS (CSE-CIC-IDS2018)](https://www.unb.ca/cic/datasets/ids-2018.html)
	- [Kaggle Labled CIC-IDS2018](https://www.kaggle.com/code/dhoogla/cse-cic-ids2018-00-cleaning/input "Kaggle Labled CIC-IDS2018")

(Note: You do not need to download the datasets if you are not planning to train any models)

## How To Develop

### Creating New Models
The following steps explain how you can create and test new models.

### Training Models
The following steps explain you can train models that exist inside the model directory.

## How To Run
### Building Virtual Environment
How to build project.

1. Run the build.py script and input the desired model option.
```
python build.py

Select one of the following models to build:
1. flaml_model

Model Number: 1
```
2. After build is activate the newly created environment.

Note: You can only have one model active at once given that each model uses a different virtual environment so you will have to build again to use a different model.

### Running
1. Ensure you have the build environement activate.

2. Run the run.py script.
```
python run.py
```

#### Usage
How to use run.py
```
usage: run.py [-h] [-t [TEST]] [-p [PREDICT]] [folder_path]

folder_path
	path to the folder to monitor for new csv files.
-p, --predict
	file path to the csv you want the model to do predictions on.
-t, --test
	test to see if build was successful and modeuls are being loaded.
```

#### Examples
Testing to see if modules are loaded successfully
```
python run.py --test
Model loaded successfully, this means build was successful!
```

Having the model predict a dataset and print out the output.
```
python run.py --predict ./dataset/test.csv
```

Monitoring a directory for new csv files. Once a new csv file is found it will automatically load it and try and predict on it.
```
python run.py ./monitor_directory
```

<!-- HOW IT WORKS -->
## How It Works
Explaining how the build process should work and functions for debugging purposes.

1. All models should be created within the models folder. Given that each model within the model folder may require different dependencies the project builds assuming that each model will use a different python virtual environment and has different requirements.

2. Build script will allow user to select a specific model to build. It will build the selected model by creating a new virtual environment and then installing the model package into the environment.

3. Once build script has completed you can activate the new environment and then run then run.py file (see Usages above on how to use).
