import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model, evaluate_classification_single
from constants import BENIGN_LABEL
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow_ranking.python.keras.metrics import MeanAveragePrecisionMetric


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df, save=False):
    print(df["label"].value_counts())

    # Set the percentage of rows to remove
    percentage_to_remove = 0.7  # Adjust this based on your requirement
    # Identify rows where 'Column2' is equal to 'benign'
    rows_to_remove = df[df[df.columns[-1]] == 'benign'].sample(frac=percentage_to_remove).index
    # Drop the identified rows
    df = df.drop(rows_to_remove)

    print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), pd.get_dummies(df[df.columns[-1]]).astype(int).to_numpy()

    X = standarize_data(X, save=save)
    # X = pca_data(X, n_components=30, save=save)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")

    # Undersampling
    # sampler = RandomUnderSampler(sampling_strategy=0.7)

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

def get_model(num_inputs, num_outputs):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=num_inputs, 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=512, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), 
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L2(1e-5)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=num_outputs, activation='softmax'),
    ])
    map = MeanAveragePrecisionMetric()
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy", map, metrics.categorical_accuracy])
    return model

def test_model(model_path):
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])

    df = clean_dataset(df)
    print(df.columns)

    # Full Classification
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=False)

    dnn_model = load_model(model_path)

    train_predictions = dnn_model.predict(X_train)
    train_predictions = np.around(train_predictions, decimals=0)

    test_predictions = dnn_model.predict(X_test)
    test_predictions = np.around(test_predictions, decimals=0)

    evaluate_classification_single(y_train, train_predictions)
    evaluate_classification_single(y_test, test_predictions)

    # evaluate_classification(dnn_model, "Traffic Classification Attack Type", X_train, X_test, y_train, y_test)


def train(save=True):
    # %%
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])

    df = clean_dataset(df)
    print(df.columns)

    # Full Classification
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=save)
    dnn_model = get_model(X_train.shape[1:], y_train.shape[-1])
    print("Model created")

    dnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=128)

    if save:
        save_model(dnn_model, "DNN", MAIN_VERSION, SAVE_FOLDER)

    evaluate_classification(dnn_model, "Traffic Classification Attack Type", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # train(save=True)
    # create_test_data()
    test_model("./saved_models/DNN_v1.1_2023-11-11.pkl")

