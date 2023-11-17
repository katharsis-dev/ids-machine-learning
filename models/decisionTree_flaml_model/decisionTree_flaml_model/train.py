# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, preprocess_attack_X, preprocess_anomaly_X, preprocess_attack_y, preprocess_anomaly_y, clean_dataset, standarize_dataset
from flaml import AutoML


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df):
    df = clean_dataset(df)

    # Get a dataframe where there are no Benign Entries so only attack types
    df_attack = df.copy()[df[df.columns[-1]] != "BENIGN"]

    # Split X and y
    X, y = preprocess_anomaly_X(df.drop(df.columns[-1], axis=1)), preprocess_anomaly_y(df[[df.columns[-1]]])

    # Split X and y attack
    X_attack, y_attack = preprocess_attack_X(df_attack.drop(df.columns[-1], axis=1)), preprocess_attack_y(df_attack[[df_attack.columns[-1]]])


    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    # Split into training and test attack
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(X_attack, y_attack, shuffle=True, test_size=0.2, random_state=42)

    # %%
    # Oversampling
    # print("Over sampling")
    sampler = RandomOverSampler(sampling_strategy="all")
    X_attack_train, y_attack_train = sampler.fit_resample(X_attack_train, y_attack_train)

    return (X_train, X_test, y_train, y_test), (X_attack_train, X_attack_test, y_attack_train, y_attack_test)


def create_test_data():
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    df = df.drop(df.columns[-1], axis=1)
    print(df.columns)
    df.to_csv("../../../datasets/test.csv", index=False)
    print("Test Dataset created")


def train(save=True):
    # %%
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")
    print(df[df.columns[-1]].value_counts())

    anomaly_data, attack_data = preprocess(df)

    # Split into training and test
    X_train, X_test, y_train, y_test = anomaly_data
    # Split into training and test attack
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = attack_data

    # %%
    # Decision Trees Anomaly Detection
    # This one is really good for Benign and not Benign
    decision_tree = DecisionTreeClassifier(max_depth=13)
    decision_tree.fit(X_train, y_train)
    if save:
        save_model(decision_tree, "decision_tree_anomaly", MAIN_VERSION, SAVE_FOLDER)
    evaluate_classification(decision_tree, "Traffic Classification", X_train, X_test, y_train, y_test)

    # Decision Trees Attack Detection
    # decision_tree_attack = DecisionTreeClassifier(max_depth=15)
    # decision_tree_attack.fit(X_attack_train, y_attack_train)
    # save_model(decision_tree_attack, "decision_tree_attack_type", MAIN_VERSION, SAVE_FOLDER)
    # evaluate_classification(decision_tree_attack, "Traffic Classification Attack", X_attack_train, X_attack_test, y_attack_train, y_attack_test)

    # %%
    # Flaml
    df = clean_dataset(df)
    # Get a dataframe where there are no Benign Entries so only attack types
    df = df.copy()[df[df.columns[-1]] != "BENIGN"]
    X, y = df.drop(df.columns[-1], axis=1).to_numpy(), df[[df.columns[-1]]].to_numpy()
    # Split into training and test attack
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    flaml_automl = AutoML()
    flaml_automl.fit(X_attack_train, y_attack_train, task="classification", time_budget=100)

    if save:
        save_model(flaml_automl, "flaml_attack_type", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_automl.best_config)
    evaluate_classification(flaml_automl, "Traffic Classification Attack", X_attack_train, X_attack_test, y_attack_train, y_attack_test)


if __name__ == "__main__":
    train(save=False)
    # create_test_data()

