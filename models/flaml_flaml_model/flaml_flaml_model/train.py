import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model, evaluate_classification_single, onehotencode_data
from constants import BENIGN_LABEL, FEATURE_SELECTION, SQL_LABEL, SSH_LABEL, FTP_LABEL, XSS_LABEL, WEB_LABEL
from flaml import AutoML


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df, save=False):
    print(df["label"].value_counts())
    df["label"] = df["label"].apply(lambda x: BENIGN_LABEL if "dos" in x else x)
    print(df["label"].value_counts())

    # Set the percentage of rows to remove
    percentage_to_remove = 0.6  # Adjust this based on your requirement
    # Identify rows where 'Column2' is equal to 'benign'
    rows_to_remove = df[df[df.columns[-1]] == 'benign'].sample(frac=percentage_to_remove).index
    # Drop the identified rows
    df = df.drop(rows_to_remove)
    print(df["label"].value_counts())

    df["label"] = df["label"].apply(lambda x: 0 if x == BENIGN_LABEL else 1)
    print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    X = standarize_data(X, save=save)
    X = pca_data(X, n_components=30, save=save)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")

    # Undersampling
    # sampler = RandomUnderSampler(sampling_strategy=0.7)

    X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def remove_rows(df, column_value, percent, random_state=42) -> pd.DataFrame:

    # Filter rows based on the condition
    mask = df[df.columns[-1]].str.contains(column_value)
    # Identify rows where 'Column2' is equal to 'benign'
    rows_to_remove = df[mask].sample(frac=percent, random_state=random_state)
    print("Removing", len(rows_to_remove))
    df = df.drop(rows_to_remove.index)

    return df

def preprocess_full(df, save=False):
    print(df["label"].value_counts())
    # Rename dos labels as benign
    df = remove_rows(df, "dos", 0.90)
    df["label"] = df["label"].apply(lambda x: BENIGN_LABEL if "dos" in x else x)
    # Rename bot labels as benign
    df = remove_rows(df, "bot", 0.70)
    df["label"] = df["label"].apply(lambda x: BENIGN_LABEL if "bot" in x else x)
    # Rename infiltration labels as benign
    df = remove_rows(df, "infiltration", 0.80)
    df["label"] = df["label"].apply(lambda x: BENIGN_LABEL if "infiltration" in x else x)

    # Rename sql label as sql-injection
    df["label"] = df["label"].apply(lambda x: SQL_LABEL if "sql" in x else x)
    # Rename ftp label as ftp-bruteforce
    df["label"] = df["label"].apply(lambda x: FTP_LABEL if "ftp" in x else x)
    # Rename ssh label as ssh-bruteforce
    df["label"] = df["label"].apply(lambda x: SSH_LABEL if "ssh" in x else x)
    # Rename xss label as xss
    df["label"] = df["label"].apply(lambda x: XSS_LABEL if "xss" in x else x)
    # Rename xss label as xss
    df["label"] = df["label"].apply(lambda x: WEB_LABEL if "web" in x else x)

    df = remove_rows(df, "benign", 0.9)
    print(df["label"].value_counts())
    # df["label"] = df["label"].apply(lambda x: 0 if x == BENIGN_LABEL else 1)
    # print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]].to_numpy().reshape(-1, 1)


    print("Columns before feature selection:", len(X.columns))
    resulting_columns = []
    for i in range(len(FEATURE_SELECTION)):
        if FEATURE_SELECTION[i]:
            resulting_columns.append(X.columns[i])
    X = X[resulting_columns]

    print("Columns after feature selection:", len(X.columns))
    X_column_names = X.columns

    X = standarize_data(X, save=save)
    # X = pca_data(X, n_components=30, save=save)

    # y = onehotencode_data(y, save=save)
    # print("One Hot Encoded Shape:", y.shape)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")

    # Undersampling
    # sampler = RandomUnderSampler(sampling_strategy=0.7)

    X_train, y_train = sampler.fit_resample(X_train, y_train)

    X_train = pd.DataFrame(X_train, columns=X_column_names)

    return X_train, X_test, y_train, y_test


def preprocess_attack_types(df, save=False):
    df = df[df["label"] != BENIGN_LABEL]
    print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    scaler = load_model("./saved_models/StandardScaler_v1.1_2023-11-11.pkl")
    X = standarize_data(X, scaler=scaler)
    pca = load_model("./saved_models/PCA_v1.1_2023-11-11.pkl")
    X = pca_data(X, n_components=30, pca=pca)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def create_test_data():
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = df.drop(df.columns[-1], axis=1)
    print(df.columns)
    df.to_csv("../../../datasets/test.csv", index=False)
    print("Test Dataset created")


def test_models():
    ANOMALY_COLUMN = "anomaly"
    ATTACK_COLUMN = "attack_type"

    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])
    df = clean_dataset(df)

    df = df.sample(n=100000)
    flaml_classification = load_model("./saved_models/flaml_classification_v1.1_2023-11-11.pkl")
    flaml_attack_classification = load_model("./saved_models/flaml_attack_type_v1.1_2023-11-11.pkl")

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
    scaler = load_model("./saved_models/StandardScaler_v1.1_2023-11-11.pkl")
    X = standarize_data(X, scaler=scaler)
    pca = load_model("./saved_models/PCA_v1.1_2023-11-11.pkl")
    X = pca_data(X, n_components=30, pca=pca)

    anomaly_predictions = flaml_classification.predict(X)

    df[ANOMALY_COLUMN] = anomaly_predictions

    attack_predictions = []
    for i in range(len(df)):
        anomaly_prediction = anomaly_predictions[i]
        attack_prediction = BENIGN_LABEL
        if anomaly_prediction == 1:
            attack_prediction = flaml_attack_classification.predict(np.array([X[i]]))
            attack_prediction = attack_prediction[0]

        attack_predictions.append(attack_prediction)

    df[ATTACK_COLUMN] = attack_predictions
    print(df[ANOMALY_COLUMN].value_counts())
    print(df[ATTACK_COLUMN].value_counts())
    print(df[ATTACK_COLUMN])
    print(y)
    evaluate_classification_single(attack_predictions, y)
    return df

def train(save=True):
    time_budget = 600
    # %%
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter/"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter/", "../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])

    df = clean_dataset(df)
    print(df.columns)

    # Flaml Classifying between benign and not
    # X_train, X_test, y_train, y_test = preprocess(df.copy(), save=save)
    # flaml_classification = AutoML()
    # flaml_classification.fit(X_train, y_train, task="classification", time_budget=time_budget)
    #
    # if save:
    #     save_model(flaml_classification, "flaml_classification", MAIN_VERSION, SAVE_FOLDER)
    #
    # print(flaml_classification.best_config)
    # evaluate_classification(flaml_classification, "Traffic Classification Attack", X_train, X_test, y_train, y_test)

    # Classifying different attack types
    # X_attack_train, X_attack_test, y_attack_train, y_attack_test = preprocess_attack_types(df.copy(), save=save)
    # flaml_attack_classification = AutoML()
    # flaml_attack_classification.fit(X_attack_train, y_attack_train, task="classification", time_budget=time_budget)
    #
    # if save:
    #     save_model(flaml_attack_classification, "flaml_attack_type", MAIN_VERSION, SAVE_FOLDER)
    #
    # print(flaml_attack_classification.best_config)
    # evaluate_classification(flaml_attack_classification, "Traffic Classification Attack Type", X_attack_train, X_attack_test, y_attack_train, y_attack_test)
    #
    # print(flaml_classification.predict(X_train[:10]))
    # print(flaml_attack_classification.predict(X_attack_train[:10]))

    # Full Classification
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = preprocess_full(df.copy(), save=save)
    print(X_attack_train.shape, y_attack_train.shape)
    print(type(X_attack_train))
    flaml_full = AutoML()

    # Because flaml is a b***h make sure you pass in pandas dataframes
    flaml_full.fit(X_attack_train, y_attack_train, task="classification", time_budget=time_budget)

    if save:
        save_model(flaml_full, "flaml_full", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_full.best_config)
    evaluate_classification(flaml_full, "Traffic Classification Attack Type", X_attack_train, X_attack_test, y_attack_train, y_attack_test)

    # print(flaml_full.predict(X_train[:10]))
    # print(flaml_full.predict(X_attack_train[:10]))

if __name__ == "__main__":
    train(save=False)
    # create_test_data()
    # test_models()

