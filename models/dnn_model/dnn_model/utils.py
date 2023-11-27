import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, mean_squared_error, max_error, multilabel_confusion_matrix


SAVE_FOLDER = "saved_models"
BENIGN_LABELS = ["BENIGN", "Benign", "benign"]
# %%

def onehotencode_data(y, encoder=None, save=False):
    if not encoder:
        encoder = OneHotEncoder()
        encoder.fit(y)
        
        if save:
            save_model(encoder, "OneHotEncoder", 1, SAVE_FOLDER, replace=True)
    y_encoded = encoder.transform(y).toarray()
    return y_encoded

def standarize_data(X, scaler=None, save=False):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)

        if save:
            save_model(scaler, "StandardScaler", 1, SAVE_FOLDER, replace=True)

    X_scaled = scaler.transform(X)
    return X_scaled

def pca_data(X, n_components=30, pca=None, save=False):
    if not pca:
        pca = PCA(n_components=n_components)
        pca.fit(X)

        if save:
            save_model(pca, "PCA", 1, SAVE_FOLDER, replace=True)

    X_reduced = pca.transform(X)
    return X_reduced

# %%
def save_model(model, name, main_version, folder_path, replace=False):
    date = datetime.now().strftime("%Y-%m-%d")
    file_name = "{0}_v{1}.{2}_{3}.pkl"

    if replace:
        file_path = os.path.join(folder_path, file_name.format(name, main_version, 1, date))
        joblib.dump(model, file_path)
        return

    for version in range(1, 100):
        check_file = os.path.join(folder_path, file_name.format(name, main_version, version, date))
        if not os.path.isfile(check_file):
            joblib.dump(model, check_file)
            return

# %%
def load_model(file_path):
    return joblib.load(file_path)

# %%
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    # Make all label values lowercase
    if ("label" in list(df.columns)):
        df["label"] = df["label"].apply(lambda x: x.lower())
        # Sort columns alphabetically but ignore the label column
        columns_to_sort = list(df.columns[:-1])
        columns_to_sort.sort()
        columns_to_sort.append(df.columns[-1])
        df = df[columns_to_sort]
    else:
        columns_to_sort = list(df.columns)
        columns_to_sort.sort()
        df = df[columns_to_sort]

    df = df.dropna()
    df = df.drop_duplicates(keep="first")
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]
    return df


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

        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True, verify_integrity=True)
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


def evaluate_classification_single(y_label, y_pred, name="Evaluation"):

    accuracy = accuracy_score(y_label, y_pred)
    
    precision = precision_score(y_label, y_pred, average="macro")
    
    recall = recall_score(y_label, y_pred, average="macro")

    f1 = f1_score(y_label, y_pred, average="macro")

    print("=" * 15, name, "=" * 15)
    print(f"Accuracy:      {round(accuracy * 100, 2):>15}")
    print(f"Precision:    {round(precision * 100, 2):>15}")
    print(f"Recall:       {round(recall * 100, 2):>15}")
    print(f"F1 Score:     {round(f1 * 100, 2):>15}")
    print(multilabel_confusion_matrix(y_label, y_pred))

# %%
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)


    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    train_precision = precision_score(y_train, train_predictions, average="macro")
    test_precision = precision_score(y_test, test_predictions, average="macro")
    
    train_recall = recall_score(y_train, train_predictions, average="macro")
    test_recall = recall_score(y_test, test_predictions, average="macro")

    train_f1 = f1_score(y_train, train_predictions, average="macro")
    test_f1 = f1_score(y_test, test_predictions, average="macro")

    
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

class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_select) -> None:
        super().__init__()
        self.features_to_select = features_to_select

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
        # print("Columns before feature selection:", len(X.columns))
        resulting_columns = []
        for i in range(len(self.features_to_select)):
            if self.features_to_select[i]:
                resulting_columns.append(X.columns[i])
        X = X[resulting_columns]
        # print("Columns after feature selection:", len(X.columns))
        # print(X.shape)
        return X


class DNNModel(BaseEstimator, TransformerMixin):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        import tensorflow as tf
        from tensorflow.keras import metrics
        from tensorflow_ranking.python.keras.metrics import MeanAveragePrecisionMetric
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu', input_shape=(num_inputs,)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=num_outputs, activation='softmax'),
            # tf.keras.layers.Dense(units=num_outputs, activation='sigmoid'),
            ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        map = MeanAveragePrecisionMetric()
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[map, metrics.categorical_accuracy, metrics.TopKCategoricalAccuracy(k=3)])
        # model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()])

    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=10, batch_size=312):
        if X_test and y_test:
            self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
        else:
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return self
        
    def transform(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)

