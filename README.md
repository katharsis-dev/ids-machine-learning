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
└── run.sh (File to run to IDS system)
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
-   [IPS/IDS dataset on AWS (CSE-CIC-IDS2018)](https://www.unb.ca/cic/datasets/ids-2018.html)
-   [Intrusion Detection Evaluation Dataset (CIC-IDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)

(Note: You do not need to download the datasets if you are not planning to train any models)

## How To Use

### Creating New Models
The following steps explain how you can create and test new models.

### Training Models
The following steps explain you can train models that exist inside the model directory.

## How To Run
### Building Virtual Environment
Run run.sh file at root level. (Currently only works on Linux :P)

Note: You can only have one model active at once given that each model uses a different virtual environment.

### Running Script
1. Go into the desired model folder and install the package into your environment.
```
pip install .
```
#### Usage
Work in Progress . . .

#### Examples
Work in Progress . . .


<!-- HOW IT WORKS -->
## How It Works (DRAFT)
Explaining how the build process should work and functions for debugging purposes.

1. All models should be created within the models folder. Given that each model within the model folder may require different dependencies the project builds assuming that each model will use a different python virtual environment.

2. Build script will go through the list of specified folders within the model folder and create the required virtual environments in order to start the build process.

3. Once virtual environments for each model have been created it will call pyinstaller on the main.py file to compile an executable file.

4. All executable files will take the data file path and the saved model path and will wither predict or evalute the model on the given data file.

5. Once all executables have been compiled build script will take all exectuables and their respective models and output the submission files for submission.
