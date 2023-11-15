# List of folder names to ignore
# FOLDER_TO_IGNORE = ['model1_brytton']
FOLDER_TO_IGNORE = ["model1_brytton"]

# BUILD_FOLDERS = ["model1_brytton"]
BUILD_FOLDERS = []

# Output Submission Directory
SUBMISSION_FOLDER = "./submissions/1"

# Group Number
GROUP_NUMBER = 1
# Arguements to use to output submissions
COMMAND_ARGUEMENTS = {
    # Submission format checkpointnumber_groupnumber_submissionnumber.csv
    "model0_template": "predict -m model.pkl -d test.csv -o ./submissions/1/1_1_1.csv",
    "model1_template": "predict -m model.pkl -d test.csv -o ./submissions/1/1_1_2.csv",
}

MODEL_FOLDER = "./models/"
