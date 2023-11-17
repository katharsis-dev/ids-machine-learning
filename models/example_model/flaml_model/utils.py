import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, mean_squared_error, max_error

# %%
def standarize_dataset(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# %%
def save_model(model, name, main_version, folder_path):
    date = datetime.now().strftime("%Y-%m-%m")
    file_name = "{0}_v{1}.{2}_{3}.pkl"
    for version in range(1, 100):
        check_file = os.path.join(folder_path, file_name.format(name, main_version, version, date))
        if not os.path.isfile(check_file):
            joblib.dump(model, check_file)
            return

# %%
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.dropna()
    df = df.drop_duplicates(keep="first")
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]
    return df

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


def load_model(file_path):
    return joblib.load(file_path)

# %%
def get_datasets_from_directory(directory_path) -> pd.DataFrame:
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
            df = pd.read_csv(file_path, low_memory=False)
        except:
            print("Failed")
            continue

        # Check if column names are consistent across CSV files
        if common_columns is None:
            common_columns = df.columns
        else:
            if not all(col in df.columns for col in common_columns):
                raise ValueError("Column names in CSV files are not consistent.")

        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        # break

    return concatenated_df

# %%
def get_dataset_from_directories(directories) -> pd.DataFrame:
    if type(directories) == list:
        dataframes = []
        for directory in directories:
            dataframes.append(get_datasets_from_directory(directory))
        return pd.concat(dataframes, ignore_index=True)
    else:
        return get_datasets_from_directory(directories)


def evalute_results(y, y_pred):
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    maxe = max_error(y, y_pred)

    print(f"Mean Absolute Error: {mae:>15}")
    print(f"Mean Squared Error: {mse:>15}")
    print(f"Max Error: {maxe:>15}")


def evaluate_regression(model, name, X, y):
    train_predictions = model.predict(X)

    mae = mean_absolute_error(train_predictions, y)

    mse = mean_squared_error(train_predictions, y)

    maxe = max_error(train_predictions, y)

    print("=" * 15, name, "=" * 15)
    print(f"Mean Absolute Error: {round(mae, 2):>20}")
    print(f"Mean Squared Error:  {round(mse, 2):>20}")
    print(f"Max Error:           {round(maxe, 2):>20}")
    print("=" * (30 + len(name) + 2))


# def evaluate_classification(model, name, X, y):
#     train_predictions = model.predict(X)
#
#
#     accuracy = accuracy_score(y, train_predictions)
#     
#     precision = precision_score(y, train_predictions)
#     
#     recall = recall_score(y, train_predictions)
#
#     f1 = f1_score(y, train_predictions)
#
#     print("=" * 15, name, "=" * 15)
#     print(f"Accuracy:      {round(accuracy * 100, 2):>15}")
#     print(f"Precision:    {round(precision * 100, 2):>15}")
#     print(f"Recall:       {round(recall * 100, 2):>15}")
#     print(f"F1 Score:     {round(f1 * 100, 2):>15}")
#     
#     confusion_matrix_result = confusion_matrix(y, train_predictions)
#     print(confusion_matrix_result)
#     print("=" * (30 + len(name) + 2), "\n")

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
