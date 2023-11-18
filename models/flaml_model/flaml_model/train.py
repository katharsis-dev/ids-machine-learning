from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification, clean_dataset, standarize_data, pca_data
from flaml import AutoML


MAIN_VERSION = 1
SAVE_FOLDER = "./saved_models/"

def preprocess(df, save=False):
    df = clean_dataset(df)

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    X = standarize_data(X, save=True)
    X = pca_data(X, n_components=30, save=True)

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
    # %%
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/filter/"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")

    X_train, X_test, y_train, y_test = preprocess(df)

    # %%
    # Flaml
    flaml_automl = AutoML()
    flaml_automl.fit(X_train, y_train, task="classification", time_budget=1000)

    if save:
        save_model(flaml_automl, "flaml", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_automl.best_config)
    evaluate_classification(flaml_automl, "Traffic Classification Attack", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    train(save=True)
    # create_test_data()

