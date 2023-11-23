from joblib import parallel_backend
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model, evaluate_classification_single
from constants import BENIGN_LABEL


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df, save=False):
    # Set the percentage of benign rows to remove
    percentage_to_remove = 0.6  # Adjust this based on your requirement
    # Identify rows where 'Column2' is equal to 'benign'
    rows_to_remove = df[df[df.columns[-1]] == 'benign'].sample(frac=percentage_to_remove).index
    # Drop the identified rows
    df = df.drop(rows_to_remove)

    df["label"] = df["label"].apply(lambda x: 0 if x == BENIGN_LABEL else 1)
    print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    X = standarize_data(X, save=save)


    X = feature_selection(X, y, n_features_to_select=20)
    print(X.head())

    X = pca_data(X, n_components=30, save=save)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")

    # Undersampling
    # sampler = RandomUnderSampler(sampling_strategy=0.7)

    X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def feature_selection(X, y, n_features_to_select=20):
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=n_features_to_select, verbose=1)
    print("Starting Feature Selection")
    with parallel_backend("threading", n_jobs=4):
        rfe.fit(X, y)
    print("Finished Fitting")

    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X.columns)]
    selected_features = [v for i, v in feature_map if i==True]
    a = [i[0] for i in feature_map]
    print(a)
    print(selected_features)
    return X.iloc[:,a]


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
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])

    df = clean_dataset(df)
    print(df.columns)

    # Full Classification
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=save)

    feature_selection(X_train, X_test, n_features_to_select=20)


    # evaluate_classification(flaml_full, "Traffic Classification Attack Type", X_attack_train, X_attack_test, y_attack_train, y_attack_test)

if __name__ == "__main__":
    train(save=True)
    # create_test_data()
    # test_models()

