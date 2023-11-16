# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
from imblearn.over_sampling import RandomOverSampler
from utils import save_model, get_dataset_from_directories, evaluate_classification
from pycaret.classification import setup
from pycaret.classification import compare_models
from flaml import AutoML


MAIN_VERSION = 1
SAVE_FOLDER = "../saved_models/"


# %%
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.dropna()
    df = df.drop_duplicates(keep="first")
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]
    return df

# %%
def standarize_dataset(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# %%
def visualize_data(df):
    import sweetviz as sv
    report_all = sv.analyze(df)
    report_all.show_html(filepath="SWEETVIZ_result.html")


def get_dnn_model(input_size, output_size):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras import metrics
    model = Sequential()
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation="softmax"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metrics.Accuracy(), metrics.Precision(), metrics.Recall(), metrics.F1Score()])

    # monitor = EarlyStopping(monitor='val_loss', patience=5)
    # model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=19)
    return model


def preprocess_anomaly_X(df_X, n_components=30):
    # Convert to numpy array
    X_anomaly = df_X.to_numpy()

    # Scale Data
    X_anomaly = standarize_dataset(X_anomaly)

    # PCA Feature and Dimentionality Reduction
    pca = PCA(n_components=n_components)
    pca = pca.fit(X_anomaly)
    X_anomaly_reduced = pca.transform(X_anomaly)
    return X_anomaly_reduced


def preprocess_attack_X(df_X):
    # Convert to numpy array
    X_attack = df_X.to_numpy()

    # Scale Data
    X_attack = standarize_dataset(X_attack)
    return X_attack

def preprocess_anomaly_y(df_y):
    # One Hot Encode the lable column and ten only take values that are Benign
    onehotencoder = OneHotEncoder()
    labels = df_y[df_y.columns[0]].values.reshape(-1, 1)
    labels = onehotencoder.fit_transform(labels).toarray()
    labels = np.logical_not(labels[:,0]).astype(float)

    # # Replace the label column with 0 and 1
    # df["label"] = labels 
    # df["label"] = df["label"].astype(float)
    return labels


def preprocess_attack_y(df_y):
    # One Hot Encode the label column values for different attack types
    onehotencoder_attack = OneHotEncoder()
    attack_labels = df_y[df_y.columns[0]].values.reshape(-1, 1)
    attack_labels = onehotencoder_attack.fit_transform(attack_labels).toarray()
    # print(attack_labels, attack_labels.shape)
    # save_model(onehotencoder_attack, "onehotencoder_attack", MAIN_VERSION, SAVE_FOLDER)
    return attack_labels


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
    # sampler = RandomOverSampler(sampling_strategy="all")
    # X_attack_train, y_attack_train = sampler.fit_resample(X_attack_train, y_attack_train)

    return (X_train, X_test, y_train, y_test), (X_attack_train, X_attack_test, y_attack_train, y_attack_test)


def create_test_data():
    df = get_dataset_from_directories(["../../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    df = df.drop(df.columns[-1], axis=1)
    df.to_csv("../../datasets/test.csv", index=False)
    print("Test Dataset created")

def train():
    # %%
    df = get_dataset_from_directories(["../../../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")
    print(df[df.columns[-1]].value_counts())
    exit()

    anomaly_data, attack_data = preprocess(df)

    # Split into training and test
    X_train, X_test, y_train, y_test = anomaly_data
    # Split into training and test attack
    X_attack_train, X_attack_test, y_attack_train, y_attack_test = attack_data

    #
    # print(np.sum(y_attack_train, axis=0))

    # %%
    # Train Model
    # models = [
    #         LinearSVC(),
    #         KNeighborsClassifier(n_jobs=-1, n_neighbors=5, leaf_size=30),
    #         DecisionTreeClassifier(max_depth=5),
    #         RandomForestClassifier(),
    #         ]

    # %%
    # KNN
    # KNN_Classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=5, leaf_size=30)
    # KNN_Classifier.fit(X_train, y_train)
    # save_model(KNN_Classifier, "../saved_models/2023-11-11_KNN_classifier_v1.pkl")

    # %%
    # Decision Trees Anomaly Detection
    # This one is really good for Benign and not Benign
    # decision_tree = DecisionTreeClassifier(max_depth=13)
    # decision_tree.fit(X_train, y_train)
    # save_model(decision_tree, "decision_tree_anomaly", MAIN_VERSION, SAVE_FOLDER)
    # evaluate_classification(decision_tree, "Traffic Classification", X_train, X_test, y_train, y_test)

    # Decision Trees Attack Detection
    # decision_tree_attack = DecisionTreeClassifier(max_depth=15)
    # decision_tree_attack.fit(X_attack_train, y_attack_train)
    # save_model(decision_tree_attack, "decision_tree_attack_type", MAIN_VERSION, SAVE_FOLDER)
    # evaluate_classification(decision_tree_attack, "Traffic Classification Attack", X_attack_train, X_attack_test, y_attack_train, y_attack_test)

    # %%
    # Deep Neural Network
    # dnn_model = get_dnn_model(X_attack_train.shape[-1], y_attack_train.shape[-1])
    # dnn_model.fit(X_attack_train, y_attack_train, validation_data=(X_attack_test, y_attack_test), epochs=10)
    # evaluate_classification(dnn_model, "DNN Model Classification", X_attack_train, X_attack_test, y_attack_train, y_attack_test)
    # print("Done Training Model!")

    # %%
    # PyCaret Model Evaluation
    # df = df.sample(frac=0.5)
    # df = clean_dataset(df)
    # label_column = df.columns[-1]
    # grid = setup(data=df, target=label_column, html=False, outliers_threshold=0.1)
    # best = compare_models()
    # print(best)

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
    # save_model(flaml_automl, "flaml_attack_type", MAIN_VERSION, SAVE_FOLDER)

    print(flaml_automl.best_config)
    evaluate_classification(flaml_automl, "Traffic Classification Attack", X_attack_train, X_attack_test, y_attack_train, y_attack_test)


if __name__ == "__main__":
    train()
    # create_test_data()

