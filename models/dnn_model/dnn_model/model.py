import numpy as np
import pandas as pd
import argparse
from .utils import load_model, clean_dataset
from .constants import SAVED_MODELS_MODULE, BENIGN_LABEL, COLUMN_LENGTH_RAW, COLUMN_LENGTH_FILTERED, REMOVE_RAW_COLUMNS, FEATURE_SELECTION
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow
import pkg_resources


class Model():
    scaler: StandardScaler
    pca: PCA

    def __init__(self, model=None, attack_model=None, pca=None, scaler=None, attack_scaler=None) -> None:

        # Load attack model
        if model is None:
            self.model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/DNN_v1.6_2023-11-11.pkl"))
        else:
            self.model = model

        # Load attack model
        if attack_model is None:
            self.attack_model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/flaml_attack_type_v1.2_2023-11-11.pkl"))
        else:
            self.attack_model = attack_model

        # Load Scaler
        if scaler is None:
            self.scaler = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/StandardScaler_v1.1_2023-11-11.pkl"))
        else:
            self.scaler = scaler

        # Load Attack Scaler
        if attack_scaler is None:
            self.attack_scaler = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/StandardScaler_Attack_v1.1_2023-11-11.pkl"))
        else:
            self.attack_scaler = attack_scaler

        # Load PCA
        if pca is None:
            self.pca = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/PCA_v1.1_2023-11-24.pkl"))
        else:
            self.pca = pca


        self.ANOMALY_CONFIDENCE = "anomaly_confidence"
        self.ANOMALY_COLUMN = "anomaly"
        self.ATTACK_COLUMN = "attack_type"

    
    def preprocess_dataframe(self, df: pd.DataFrame):
        """
        Clean up the dataframe before passing into model for predictions
        """
        X = df
        resulting_columns = []
        for i in range(len(FEATURE_SELECTION)):
            if FEATURE_SELECTION[i]:
                resulting_columns.append(X.columns[i])
        X = X[resulting_columns]
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def preprocess_attack_dataframe(self, df: pd.DataFrame):
        X = self.attack_scaler.transform(df)
        X = self.pca.transform(X)
        return X


    def predict(self, df: pd.DataFrame):
        """
        Predict outputs based on the given data
        """
        df = clean_dataset(df)

        if len(df.columns) == COLUMN_LENGTH_FILTERED:
            df_filtered = df
        # else len(df.columns) == COLUMN_LENGTH_RAW:
        else:
            original_columns_set = set(df.columns)
            new_columns_set = list(original_columns_set - set(REMOVE_RAW_COLUMNS))
            new_columns_set.sort()
            df_filtered = df[new_columns_set]

        X = self.preprocess_dataframe(df_filtered)
        X_attack = self.preprocess_attack_dataframe(df_filtered)

        predictions = self.model.predict(X)
        df[self.ANOMALY_CONFIDENCE] = predictions
        df[self.ANOMALY_COLUMN] = [1 if prediction > 0.5 else 0 for prediction in predictions]

        attack_type_result = []
        anomaly_values = df[self.ANOMALY_COLUMN].tolist()
        for index in range(len(anomaly_values)):
            result_label = BENIGN_LABEL
            anomaly_value = anomaly_values[index]
            if anomaly_value == 1:
                result_label = self.attack_model.predict(np.array([X_attack[index]]))
                result_label = result_label[0]
            attack_type_result.append(result_label)

        df[self.ATTACK_COLUMN] = attack_type_result
        # print(df[self.ANOMALY_COLUMN].value_counts())
        print(df[self.ATTACK_COLUMN].value_counts())
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a file.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    input_file_path = args.input_file
    output_file_path = args.output_file

    model = Model()

    # df = pd.read_csv("../../datasets/test.csv")
    df = pd.read_csv(input_file_path)
    result = model.predict(df)
    result.to_csv(output_file_path)
