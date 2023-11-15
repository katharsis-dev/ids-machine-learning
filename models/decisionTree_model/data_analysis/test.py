# %%
import autosklearn.classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
from imblearn.over_sampling import RandomOverSampler
import pickle5 as pickle

# %%
def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
        return model

# %%
def get_datasets_from_directory(directory_path):
    # Check if the given path is a directory
    if not os.path.isdir(directory_path):
        raise ValueError("Provided path is not a directory.")

    # Get a list of all files in the directory with .csv extension
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    # Check if there are any CSV files in the directory
    if not csv_files:
        raise ValueError("No CSV files found in the given directory.")

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    # Initialize a variable to store the common column names
    common_columns = None

    # Concatenate each CSV file into the DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        try:
            print(file_path)
            df = dd.read_csv(file_path, low_memory=False)
        except:
            print("Failed")
            continue

        # Check if column names are consistent across CSV files
        if common_columns is None:
            common_columns = df.columns
        else:
            if not all(col in df.columns for col in common_columns):
                raise ValueError("Column names in CSV files are not consistent.")

        concatenated_df = dd.concat([concatenated_df, df], ignore_index=True)
        # break

    return concatenated_df

# %%
def get_dataset_from_directories(directories):
    if type(directories) == list:
        dataframes = []
        for directory in directories:
            dataframes.append(get_datasets_from_directory(directory))
        return dd.concat(dataframes, ignore_index=True)
    else:
        return get_datasets_from_directory(directories)


# %%
def clean_dataset(df):
    # assert isinstance(df, dd.DataFrame), "df needs to be a pd.DataFrame"
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df.dropna()
    # df.drop_duplicates(keep="first")
    # indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
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

# %%
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)


    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    train_precision = precision_score(y_train, train_predictions, average="micro")
    test_precision = precision_score(y_test, test_predictions, average="micro")
    
    train_recall = recall_score(y_train, train_predictions, average="micro")
    test_recall = recall_score(y_test, test_predictions, average="micro")

    train_f1 = f1_score(y_train, train_predictions, average="micro")
    test_f1 = f1_score(y_test, test_predictions, average="micro")

    
    print("Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    print("Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100))
    print("Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100))
    print("Training F1 " + str(name) + " {}  Test F1 ".format(train_f1) + str(name) + " {}".format(test_f1))
    
    actual = y_test
    predicted = model.predict(X_test)
    try:
        confusion_matrix_result = confusion_matrix(actual, predicted)
        print(confusion_matrix_result)
    except ValueError:
        print("Confusion Matrix Only supported for binary")

    # cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['normal', 'attack'])
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.grid(False)
    # cm_display.plot(ax=ax)

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



# %%
# df = get_dataset_from_directories(["../../datasets/CIC-IDS-2017/MachineLearningCVE/"])
df = get_dataset_from_directories(["../../datasets/CSE-CIC-IDS2018/"])
# df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
# df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")

# %%
# For Visualizing the data
# visualize_data(df)

# %%
df = clean_dataset(df)

print(df.head())
print(df.columns, len(df), len(df.columns))
print(df["label"].unique())
print(df["label"].value_counts())
