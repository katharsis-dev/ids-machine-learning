from constants import IMPORT_PACKAGE
import os
import time
import argparse
import pandas as pd

# Dynamic Import for loaded model
try:
    file = open(IMPORT_PACKAGE, "r")
    import_command = file.read().strip()
    exec(import_command)
    model = Model()
except Exception as e:
    print(f"Error failed to import model package: {e}")

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

def get_new_csv_files(folder_path, last_seen_files):
    all_files = set(os.listdir(folder_path))
    new_files = all_files - last_seen_files
    return new_files

def process_new_csv_files(folder_path, new_files):
    for file in new_files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            # Process the DataFrame as needed
            print(f"New CSV file '{file}' has been added:")
            print(df)
            print("\n")

def predict_function(file_path):
    # print(f"Running test function with argument: {test_arg}")
    df = pd.read_csv(file_path)
    result = model.predict(df)
    print(result)

def test_function():
    print(f"Model loaded successfully, this means build was successful!")

def main():
    parser = argparse.ArgumentParser(description="Monitor a folder for new CSV files.")
    parser.add_argument("-t", "--test", nargs='?', const=True, help="Run a test function with an optional argument.")
    parser.add_argument("-p", "--predict", nargs='?', const=True, help="Get predictions given path to a dataset.")
    parser.add_argument("folder_path", nargs='?', type=str, help="Path to the folder to monitor (optional if --test is present).")
    args = parser.parse_args()

    if args.test:
        if args.folder_path:
            print("Warning: Ignoring folder_path argument because --test flag is present.")
        test_function()
        return

    if args.predict:
        if args.folder_path:
            print("Warning: Ignoring folder_path argument because --test flag is present.")
        predict_function(args.predict)
        return

    folder_path = args.folder_path
    if not folder_path:
        parser.error("The folder_path argument is required when --test or --predict flag is not present.")
    check_folder(folder_path)

    last_seen_files = set()

    try:
        while True:
            new_files = get_new_csv_files(folder_path, last_seen_files)
            if new_files:
                process_new_csv_files(folder_path, new_files)
                last_seen_files.update(new_files)

            time.sleep(5)

    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    main()
