from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
from imblearn.over_sampling import RandomOverSampler


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

    return concatenated_df

def get_dataset_from_directories(directories):
    if type(directories) == list:
        dataframes = []
        for directory in directories:
            dataframes.append(get_datasets_from_directory(directory))
        return pd.concat(dataframes, ignore_index=True)
    else:
        return get_datasets_from_directory(directories)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]

def standarize_dataset(df):
    column_names = df.columns.tolist()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(data=df_scaled, columns=column_names)
    return df_scaled

def visualize_data(df):
    import sweetviz as sv
    report_all = sv.analyze(df)
    report_all.show_html(filepath="SWEETVIZ_result.html")

def preprocess(df):
    label_column = "label"

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    df = clean_dataset(df)


    print(df.head())
    print(df.columns, len(df), len(df.columns))
    print(df["label"].unique())
    print(df["label"].value_counts())


    # This is to OneHotEncode the labels and then only keep the BENIGN
    onehotencoder = OneHotEncoder()
    labels = df["label"].values.reshape(-1, 1)
    labels = onehotencoder.fit_transform(labels).toarray()
    encoded_labels = labels[:,0].shape

    df["label"] = encoded_labels
    print(df["label"].value_counts())
    exit()

    X, y = df.drop(label_column, axis=1).to_numpy(), df[[label_column]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)


    sampler = RandomOverSampler(sampling_strategy="all")
    X_train, y_train = sampler.fit_resample(X, y)


    # df_no_label = df_clean.drop("label", axis=1, inplace=False)
    # df_clean_scaled = standarize_dataset(df_no_label)
    #

    return X_train, X_test, y_train, y_test

def principleComponentAnalysis(X, n):
    pca = PCA(n_components=n)
    pca = pca.fit(X)
    return pca.transform(X)


if __name__ == "__main__":
    # df = get_dataset_from_directories(["../datasets/CSE-CIC-IDS2018/"])
    df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = get_dataset_from_directories(["../datasets/CIC-IDS-2017/TrafficLabelling/", "../datasets/CIC-IDS-2017/MachineLearningCVE/"])
    # df = pd.read_csv("../datasets/CIC-IDS-2017/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")

    X_train, X_test, y_train, y_test = preprocess(df)

    # For Visualizing the data
    # visualize_data(df)
