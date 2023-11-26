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

    def __init__(self, model=None, attack_model=None, pca=None, scaler=None, attack_scaler=None, encoder=None) -> None:

        # Load attack model
        if model is None:
            self.model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/DNN_v1.1_2023-11-25.pkl"))
        else:
            self.model = model

        # Load Scaler
        if scaler is None:
            self.scaler = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/StandardScaler_v1.1_2023-11-25.pkl"))
        else:
            self.scaler = scaler

        # Load Encoder
        if encoder is None:
            self.encoder = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/OneHotEncoder_v1.1_2023-11-25.pkl"))
        else:
            self.encoder = encoder

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

        predictions = self.model.predict(X)

        # Find the index of the maximum value in each row
        max_indices = np.argmax(predictions, axis=1)
        # Create a one-hot encoded representation
        one_hot_encoded = np.zeros_like(predictions)
        one_hot_encoded[np.arange(len(predictions)), max_indices] = 1

        predictions = self.encoder.inverse_transform(one_hot_encoded)

        df[self.ATTACK_COLUMN] = predictions.flatten()

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
