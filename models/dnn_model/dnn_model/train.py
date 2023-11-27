import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model, evaluate_classification_single, onehotencode_data
import utils
from constants import BENIGN_LABEL, FEATURE_SELECTION, SQL_LABEL, SSH_LABEL, FTP_LABEL, XSS_LABEL, WEB_LABEL
import tensorflow as tf
from sklearn.pipeline import Pipeline
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow_ranking.python.keras.metrics import MeanAveragePrecisionMetric


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"


def preprocess_labels(df_X, df_y, save=False):
    def remove_rows(df_X, df_y, column_value, percent, random_state=42) -> tuple:

        # Filter rows based on the condition
        mask = df_y[df_y.columns[-1]].str.contains(column_value)
        # Identify rows where 'Column2' is equal to 'benign'
        rows_to_remove = df_y[mask].sample(frac=percent, random_state=random_state)
        print("Removing", len(rows_to_remove))
        df_y = df_y.drop(rows_to_remove.index)
        df_X = df_X.drop(rows_to_remove.index)

        return df_X, df_y

    print(df_y["label"].value_counts())
    # Rename dos labels as benign
    df_X, df_y = remove_rows(df_X, df_y, "dos", 0.90)
    df_y["label"] = df_y["label"].apply(lambda x: BENIGN_LABEL if "dos" in x else x)
    # Rename bot labels as benign
    df_X, df_y = remove_rows(df_X, df_y, "bot", 0.70)
    df_y["label"] = df_y["label"].apply(lambda x: BENIGN_LABEL if "bot" in x else x)
    # Rename infiltration labels as benign
    df_X, df_y = remove_rows(df_X, df_y, "infiltration", 0.80)
    df_y["label"] = df_y["label"].apply(lambda x: BENIGN_LABEL if "infiltration" in x else x)

    # Rename sql label as sql-injection
    df_y["label"] = df_y["label"].apply(lambda x: SQL_LABEL if "sql" in x else x)
    # Rename ftp label as ftp-bruteforce
    df_y["label"] = df_y["label"].apply(lambda x: FTP_LABEL if "ftp" in x else x)
    # Rename ssh label as ssh-bruteforce
    df_y["label"] = df_y["label"].apply(lambda x: SSH_LABEL if "ssh" in x else x)
    # Rename xss label as xss
    df_y["label"] = df_y["label"].apply(lambda x: XSS_LABEL if "xss" in x else x)
    # Rename xss label as xss
    df_y["label"] = df_y["label"].apply(lambda x: WEB_LABEL if "web" in x else x)

    df_X, df_y = remove_rows(df_X, df_y, "benign", 0.9)
    print(df_y["label"].value_counts())

    return df_X, df_y


def preprocess(df, save=False):

    def remove_rows(df, column_value, percent, random_state=42) -> pd.DataFrame:
        # Filter rows based on the condition
        mask = df[df.columns[-1]].str.contains(column_value)
        # Identify rows where 'Column2' is equal to 'benign'
        rows_to_remove = df[mask].sample(frac=percent, random_state=random_state)
        print("Removing", len(rows_to_remove))
        df = df.drop(rows_to_remove.index)

        return df

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

    X = standarize_data(X, save=save)
    # X = pca_data(X, n_components=30, save=save)

    y = onehotencode_data(y, save=save)
    print("One Hot Encoded Shape:", y.shape)

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
        tf.keras.layers.Dense(units=128, activation='relu', input_shape=num_inputs),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_outputs, activation='softmax'),
        # tf.keras.layers.Dense(units=num_outputs, activation='sigmoid'),
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    map = MeanAveragePrecisionMetric()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[map, metrics.categorical_accuracy, metrics.TopKCategoricalAccuracy(k=3)])
    # model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()])
    return model

def test_model(model_path):
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter", "../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/Custom/labeled/filter/"])

    df = clean_dataset(df)
    print(df.columns)

    # Full Classification
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=False)
    print(X_train.shape, y_train.shape)

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
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter", "../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/Custom/filter/"])

    df = clean_dataset(df)

    # Full Classification
    X_train, X_test, y_train, y_test = preprocess(df.copy(), save=save)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    dnn_model = get_model(X_train.shape[1:], y_test.shape[-1])
    print("Model created")
    return
    dnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=312)
    if save:
        save_model(dnn_model, "DNN", MAIN_VERSION, SAVE_FOLDER)

    train_predictions = dnn_model.predict(X_train)
    train_predictions = np.around(train_predictions, decimals=0)

    test_predictions = dnn_model.predict(X_test)
    test_predictions = np.around(test_predictions, decimals=0)

    evaluate_classification_single(y_train, train_predictions)
    evaluate_classification_single(y_test, test_predictions)

def run_pipeline(save=True):
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter", "../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/Custom/labeled/filter/"])
    df = clean_dataset(df)

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1:]]
    X, y = preprocess_labels(X, y)

    y = onehotencode_data(y, save=save)
    print("One Hot Encoded Shape:", y.shape)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")
    # Undersampling
    # sampler = RandomUnderSampler(sampling_strategy=0.7)
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    dnn_model = utils.DNNModel(20, 8)
    pipeline = Pipeline([
        ("feature_selection", utils.FeatureSelection(FEATURE_SELECTION)),
        ("standard_scaler", utils.StandarizeData()),
        ("dnn_model", dnn_model),
        ])
    pipeline.fit(X_train, y_train)

    save_model(pipeline, "DNN_Pipeline", MAIN_VERSION, SAVE_FOLDER)

    train_predictions = pipeline.predict(X_train)
    train_predictions = np.around(train_predictions, decimals=0)

    test_predictions = pipeline.predict(X_test)
    test_predictions = np.around(test_predictions, decimals=0)

    evaluate_classification_single(y_train, train_predictions)
    evaluate_classification_single(y_test, test_predictions)

def evaluate_pipeline(pipeline):
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter", "../../../datasets/Custom/labeled/filter/"])
    # df = get_dataset_from_directories(["../../../datasets/Custom/labeled/filter/"])
    df = clean_dataset(df)

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1:]]
    X, y = preprocess_labels(X, y)

    y = onehotencode_data(y, save=False)
    print("One Hot Encoded Shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    train_predictions = pipeline.predict(X_train)
    train_predictions = np.around(train_predictions, decimals=0)

    test_predictions = pipeline.predict(X_test)
    test_predictions = np.around(test_predictions, decimals=0)

    evaluate_classification_single(y_train, train_predictions)
    evaluate_classification_single(y_test, test_predictions)

def save_pipeline(pipeline):
    # ("feature_selection", utils.FeatureSelection(FEATURE_SELECTION)),
    # ("standard_scaler", utils.StandarizeData()),
    # ("dnn_model", dnn_model),
    feature_selection = pipeline.named_steps["feature_selection"]
    standard_scaler = pipeline.named_steps["standard_scaler"]
    dnn_model = pipeline.named_steps["dnn_model"]
    save_model(dnn_model, "TESTING", MAIN_VERSION, SAVE_FOLDER)
    print(type(dnn_model))

if __name__ == "__main__":
    # run_pipeline(save=True)

    # evaluate_pipeline(load_model("./saved_models/DNN_Pipeline_v1.1_2023-11-26.pkl"))
    save_pipeline(load_model("./saved_models/DNN_Pipeline_v1.1_2023-11-26.pkl"))
    # train(save=False)
    # create_test_data()
    # test_model("./saved_models/DNN_v1.1_2023-11-25.pkl")

