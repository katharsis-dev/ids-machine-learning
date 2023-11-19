from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data, load_model
from constants import BENIGN_LABEL
from flaml import AutoML


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df, save=False):
    df["label"] = df["label"].apply(lambda x: x.lower())
    df["label"] = df["label"].apply(lambda x: 0 if x == BENIGN_LABEL else 1)
    print(df["label"].value_counts())

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    X = standarize_data(X, save=True)
    X = pca_data(X, n_components=30, save=True)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Oversampling
    sampler = RandomOverSampler(sampling_strategy="all")
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def preprocess_attack_types(df, save=False):
    df["label"] = df["label"].apply(lambda x: x.lower())
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


def train(save=True):
    time_budget = 500
    # %%
    # df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/", "../../../datasets/CIC-IDS-2018/filter"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")
    df = clean_dataset(df)

    # Flaml Classifying between benign and not
    X_train, X_test, y_train, y_test = preprocess(df.copy())
    flaml_classification = AutoML()
    flaml_classification.fit(X_train, y_train, task="classification", time_budget=time_budget)

    if save:
        save_model(flaml_classification, "flaml_classification", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_classification.best_config)
    evaluate_classification(flaml_classification, "Traffic Classification Attack", X_train, X_test, y_train, y_test)

    # Classifying different attack types
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = preprocess_attack_types(df.copy())
    flaml_attack_classification = AutoML()
    flaml_attack_classification.fit(X_attack_train, y_attack_train, task="classification", time_budget=time_budget)

    if save:
        save_model(flaml_attack_classification, "flaml_attack_type", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_attack_classification.best_config)
    evaluate_classification(flaml_attack_classification, "Traffic Classification Attack Type", X_attack_train, X_attack_test, y_attack_train, y_attack_test)

    # print(flaml_classification.predict(X_train[:10]))
    # print(flaml_attack_classification.predict(X_attack_train[:10]))

if __name__ == "__main__":
    train(save=True)
    # create_test_data()

