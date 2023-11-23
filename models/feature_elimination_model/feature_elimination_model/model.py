import numpy as np
import pandas as pd
import argparse
from .utils import load_model, clean_dataset
from .constants import SAVED_MODELS_MODULE, BENIGN_LABEL, COLUMN_LENGTH_RAW, COLUMN_LENGTH_FILTERED, REMOVE_RAW_COLUMNS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pkg_resources


class Model():
    scaler: StandardScaler
    pca: PCA

    def __init__(self, model, pca=None, scaler=None) -> None:

        # Load attack model
        if model is None:
            self.model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/flaml_full_v1.1_2023-11-11.pkl"))
        else:
            self.model = model

        # Load Scaler
        if scaler is None:
            self.scaler = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/StandardScaler_v1.1_2023-11-11.pkl"))
        else:
            self.scaler = scaler

        # Load PCA
        if pca is None:
            self.pca = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/PCA_v1.1_2023-11-11.pkl"))
        else:
            self.pca = pca


        self.ANOMALY_COLUMN = "anomaly"
        self.ATTACK_COLUMN = "attack_type"

    
    def preprocess_dataframe(self, df: pd.DataFrame):
        """
        Clean up the dataframe before passing into model for predictions
        """
        X = self.scaler.transform(df)
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

        predictions = self.model.predict(X)
        df[self.ANOMALY_COLUMN] = predictions
        print(df[self.ANOMALY_COLUMN].value_counts())
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
