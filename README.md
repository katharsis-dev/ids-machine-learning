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
│   ├── decisionTree_flaml_model/
│   │   ├── decisionTree_flaml_model/
│   │   │   ├── model.py
│   │   │   ├── utils.py
│   │   │   └── train.py
│   │   ├── saved_models/
│   │   ├── requirements.txt
│   │   └── setup.py
│   └── ...
├── tools (Data analysis tools and scripts)/
│   ├── data_analysis/ (Scripts for data analysis)
│   └── data_filter/ (Scripts for formatting dataset for training)/
│       ├── filter_cic-ids-2017.py
│       └── filter_cic-ids-2018.py
├── requirements.txt (Project Dependencies)
├── README.md (Read me file)
├── constants.py (Variables used in build.py for configuration)
├── build.py (File for building project)
└── run.py (File to run to start IDS system)
```

<!-- GETTING STARTED -->
## Datasets
List of datasets used:
-   [Intrusion Detection Evaluation Dataset (CIC-IDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)
-   [IPS/IDS dataset on AWS (CSE-CIC-IDS2018)](https://www.unb.ca/cic/datasets/ids-2018.html)
	- [Kaggle Labled CIC-IDS2018](https://www.kaggle.com/code/dhoogla/cse-cic-ids2018-00-cleaning/input "Kaggle Labled CIC-IDS2018")

(Note: You do not need to download the datasets if you are not planning to train any models)

### Formatting & Filtering Dataset
After you download the dataset run the scripts within the tools/data_filter/ folder in order to format them for training. This is only required if you want to train the models.

## How To Develop
### Prerequisites
#### Python
List of prerequisites that should be installed before hand.

1. Have Python installed with pip to create virtual environments.
https://www.python.org/downloads/

2. Create base virtual environment with the requirements.txt file at the root level.


### Creating New Models
The following steps explain how you can create and test new models.

1. Copy one of the folders within the models folder as a template (ex. /models/flaml_model).
2. Rename the copied folder to the following format **[model type]_model**.
3. Rename the folder within the newly created folder to match your new folder name.
4. Edit the setup.py script to match the new folder name. This is so that pip will know what folder to use for installing the package.
5. Edit the train.py for your specific model. Train your model by running the script and then save the models with in the saved_models folder.
6. Edit model.py so that it loads your saved models and then also preprocesses data to fit your model input.
7. Create a new requirements.txt for your environment which should include any new modules you have installed.
8. Now you can run build.py to build your model.
9. See "How To Run" to run your newly built model.

### Training Models
The following steps explain you can train models that exist inside the model directory.

1. To train models run the train.py script for that specific model located with in the models folder.
2. Make sure you have setup an environment with all the required dependencies to run the train.py script.
3. Models should be saved within the saved_models folder and you will need to update the model.py script to load the new models instaed.

## How To Run
How to run the the intrusion detection system on your computer.

### Building
1. Build the project by running the build.py script and select the desired model. Make sure you have python installed.
```
python build.py

Select one of the following models to build:
1. flaml_model
2. example

Model Number: 1
```
2. Once build is complete you will need to activate the newly created environment. Path to the new environment should be printed in the terminal.
3. Now you can run the run.py script with the newly created environment to start monitoring and detecting network traffic.

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
