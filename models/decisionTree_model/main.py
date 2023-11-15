import sklearn
import pandas as pd
from utils import load_model
from train import clean_dataset, preprocess_attack_X, preprocess_anomaly_X

class DecisionTreeModel():


    def __init__(self, anomaly_model, attack_model) -> None:
        self.anomaly_model = anomaly_model
        self.attack_model = attack_model
    
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

        X = df.to_numpy()

        X_anomaly = preprocess_anomaly_X(X)
        X_attack = preprocess_attack_X(X)

        anomaly_predictions = self.anomaly_model.predict(X_anomaly)


if __name__ == "__main__":
    anomaly_model = load_model("./saved_models/decision_tree_suspicious_v1.1_2023-11-11.pkl")
    attack_model = load_model("./saved_models/decision_tree_attack_type_v1.1_2023-11-11.pkl")

    model = DecisionTreeModel(anomaly_model, attack_model)
