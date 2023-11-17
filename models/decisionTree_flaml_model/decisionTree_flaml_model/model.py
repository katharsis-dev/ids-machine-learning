import numpy as np
import pandas as pd
import argparse
from .utils import load_model, clean_dataset, preprocess_attack_X, preprocess_anomaly_X
from .constants import SAVED_MODELS_MODULE
import pkg_resources


class Model():

    def __init__(self, anomaly_model=None, attack_model=None) -> None:

        # Load anomaly model
        if anomaly_model is None:
            self.anomaly_model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/decision_tree_anomaly_v1.1_2023-11-11.pkl"))

        else:
            self.anomaly_model = anomaly_model

        # Load attack model
        if attack_model is None:
            self.attack_model = load_model(pkg_resources.resource_filename(__package__, f"{SAVED_MODELS_MODULE}/flaml_attack_type_v1.1_2023-11-11.pkl"))
        else:
            self.attack_model = attack_model


        self.ANOMALY_COLUMN = "anomaly"
        self.ATTACK_COLUMN = "attack_type"

    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the dataframe before passing into model for predictions
        """
        return clean_dataset(df)


    def predict(self, df: pd.DataFrame):
        """
        Predict outputs based on the given data
        """
        df = self.preprocess_dataframe(df)

        X_anomaly = preprocess_anomaly_X(df)
        # X_attack = preprocess_attack_X(df)
        # X_attack = preprocess_attack_X(df)
        X_attack = df.to_numpy()

        anomaly_predictions = self.anomaly_model.predict(X_anomaly)

        df[self.ANOMALY_COLUMN] = anomaly_predictions

        attack_predictions = []
        for i in range(len(df)):
            anomaly_prediction = anomaly_predictions[i]
            attack_prediction = "BENIGN"
            if anomaly_prediction == 1:
                attack_prediction = self.attack_model.predict(np.array([X_attack[i]]))
                attack_prediction = attack_prediction[0]

            attack_predictions.append(attack_prediction)

        df[self.ATTACK_COLUMN] = attack_predictions
        # print(df[self.ATTACK_COLUMN].value_counts())
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
