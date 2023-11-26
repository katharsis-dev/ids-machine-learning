import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model, evaluate_classification_single
from constants import BENIGN_LABEL
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

def preprocess_full(df, save=False):
    print(df["label"].value_counts())

    # Set the percentage of rows to remove
    percentage_to_remove = 0.7  # Adjust this based on your requirement
    # Identify rows where 'Column2' is equal to 'benign'
    rows_to_remove = df[df[df.columns[-1]] == 'benign'].sample(frac=percentage_to_remove).index
    # Drop the identified rows
    df = df.drop(rows_to_remove)

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
    time_budget = 400
    # %%
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter/", "../../../datasets/Custom/filter/"])
    df = get_dataset_from_directories(["../../../datasets/Custom/filter/"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])

    df = clean_dataset(df)
    print(df.columns)

    # Flaml Classifying between benign and not
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=save)
    flaml_classification = AutoML()
    flaml_classification.fit(X_train, y_train, task="classification", time_budget=time_budget)

    if save:
        save_model(flaml_classification, "flaml_classification", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_classification.best_config)
    evaluate_classification(flaml_classification, "Traffic Classification Attack", X_train, X_test, y_train, y_test)

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
    # X_attack_train, X_attack_test, y_attack_train, y_attack_test = preprocess_full(df.copy(), save=save)
    # flaml_full = AutoML()
    # flaml_full.fit(X_attack_train, y_attack_train, task="classification", time_budget=time_budget)
    #
    # if save:
    #     save_model(flaml_full, "flaml_full", MAIN_VERSION, SAVE_FOLDER)
    #
    # print(flaml_full.best_config)
    # evaluate_classification(flaml_full, "Traffic Classification Attack Type", X_attack_train, X_attack_test, y_attack_train, y_attack_test)
    #
    # print(flaml_full.predict(X_train[:10]))
    # print(flaml_full.predict(X_attack_train[:10]))

if __name__ == "__main__":
    train(save=False)
    # create_test_data()
    # test_models()

