<!-- PROJECT LOGO -->
# COMP-4983 Insurance Claim Prediction
<!-- ABOUT THE PROJECT -->
## About The Project

Goal of this project is to perform data exploration and preprocessing on unlabled data through multiple iterations to test and evaluate different models.

## Project Structure

```
├── dataset (Dataset used)/
│   ├── test.csv
│   └── trainingset.csv
├── submissions/
│   ├── 1/
│   │   ├── 1_1_1.csv
│   │   ├── 1_1_2.csv
│   │   └── ...
│   ├── 2/
│   │   └── ...
│   ├── 3
│   └── 4
├── models (Models used for prediction)/
│   ├── model0_template/
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   └── utils.py
│   ├── model1_KNN_brytton/
│   │   ├── requirements.txt
│   │   └── main.py
│   ├── model2_DNN_brytton/
│   │   ├── requirements.txt
│   │   └── main.py
│   └── model...
├── build.py
├── build_configs.py (Configurations for Building)
├── run.py
├── requirements.txt (Project Dependencies)
└── README.md (Read me file)
```

## Submissions

### Checkpoint #1 
```
1_1_1.csv MAE = 9086.27
1_1_2.csv MAE = 195.72
1_1_3.csv MAE = 168.46
1_1_4.csv MAE = 99.93
1_1_5.csv MAE = 291.84  # Deep neural network regression
1_1_6.csv MAE = 117.68  # XGBoost regression
1_1_7.csv MAE = 105.88  # RandomForestClassifier + ExtraTreesRegression
```

<!-- GETTING STARTED -->
## How To Use
### Prerequisites
List of prerequisties that should be installed before hand.

1. Have Python insatlled with pip to create virual environments.
https://www.python.org/downloads/

### Creating New Models
The following steps explain how you can create and test new models.

1. Create a new directory under models. Please follow the naming scheme **"model[##]_[model_type]_[your name]_[optional description]"**.

2. Setup your virtual environment as desired and create a [requirements.txt](http://https://stackoverflow.com/questions/31684375/automatically-create-file-requirements-txt "requirements.txt").

	**Recommended:** Create your virtual environments within your model directory so it is easy for you activate in the future and you don't get confused
```
python -m venv /models/model0_example/venv
```

3. Activate your newly created [virual environment]( https://python.land/virtual-environments/virtualenv "virual environment").
```
# Linux
source /models/model0_example/venv/bin/activate

# Windows
/models/model0_example/venv/bin/activate
powershell /models/model0_example/venv/bin/activate.ps1
```

4. Install pyinstaller `pip install pyinstaller` for compiling code into executable

5. Copy the the /models/model0_template/main.py file as the entry point for your code. (This is so your code can be compiled into an executable later on)

6. Modify predict function in the main.py file so that it fits your preprocessing and outputs a csv file in the submission format.

### How To Run
How to run main.py or executable after compiling.

### How To Build

1. Run the build.py Python script.

#### Usage
```
usage: main [-h] {predict,evaluate} -d DATASET_PATH -m MODEL_PATH [-o OUTPUT_PATH]

command
	predict
        Predict outputs based on the given dataset and model.
	evaluate
        Evaluate model based on the given dataset and model.
-d, --dataset
	path to the dataset to load.
-m, --model
	path to the model to load.
-o, --output
	Path the output the predictions.

```
#### Examples
```
./dist/main predict -d ../../dataset/testset.csv -m ./saved_models/linear_model.pkl -o result.csv

# result.csv
rowIndex,ClaimAmount
0,42.258355753841826
1,95.63259244898056
2,196.71112271535935
3,126.66342615445835
```


<!-- HOW IT WORKS -->
## How It Works (DRAFT)
Explaining how the build process should work and fucntion for debugging purposes.

1. All models should be created within the models folder. Given that each model within the model folder may require different dependencies the project builds assuming that each mode will use a different python virtual environment.

2. Build script will go through the list of specified folders within the model folder and create the required virtual environments in order to start the build process.

3. Once virtual environments for each model have been created it will call pyinstaller on the main.py file to compile an executable file.

4. All executable files will take the data file path and the saved model path and will wither predict or evalute the model on the given data file.

5. Once all executables have been compiled build script will take all exectuables and their respective models and output the submission files for submission.

<!-- Submission -->
## How to Submit (IN PROGRESS)
Work in progress. . .

Still need to create a build script :(
