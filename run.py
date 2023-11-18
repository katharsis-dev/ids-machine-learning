import os
import time
import argparse
import platform
import subprocess
from numpy import column_stack
import pandas as pd
import datetime
from constants import IMPORT_PACKAGE

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Dynamic Import for loaded model
try:
    file = open(IMPORT_PACKAGE, "r")
    import_command = file.read().strip()
    exec(import_command)
    model = Model()
except Exception as e:
    print(f"Error failed to import model package: {e}")
    exit(1)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")


def get_new_csv_files(folder_path, last_seen_files):
    all_files = set(os.listdir(folder_path))
    new_files = all_files - last_seen_files
    return new_files


def convert_pcap_to_csv(pcap_file_path, csv_file_path):
    command = f"cicflowmeter -f {pcap_file_path} -c {csv_file_path}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)


def predict_function(file_path):
    # print(f"Running test function with argument: {test_arg}")
    df = pd.read_csv(file_path)
    result = model.predict(df)
    print(result)


def test_function():
    print(f"Model loaded successfully, this means build was successful!")


def monitor_folder(folder_path, delay=5):
    last_seen_files = set()
    print(f"Started Monitoring {folder_path}")
    while True:
        new_files = get_new_csv_files(folder_path, last_seen_files)
        if new_files:
            process_new_csv_files(folder_path, new_files)
            last_seen_files.update(new_files)
        else:
            print("No New Files")

        time.sleep(delay)


def check_tcp_dump():
    if not check_command_available("tcpdump"):
        print("Missing tcpdump Command, please insatll with (apt-get install tcpdump)")
        exit(1)
    else:
        print("tcpdump command found")

        
def check_command_available(command):
    try:
        # Use 'where' on Windows, 'which' on Unix-like systems
        if platform.system().lower() == 'windows':
            subprocess.run(['where', command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            subprocess.run(['which', command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def process_new_csv_files(folder_path, new_files):
    """
    Edit this function to change prediction reuslt behaviour
    """
    for file in new_files:
        if file.endswith('.pcap'):
            pcap_file_path = os.path.join(folder_path, file)
            csv_file_path = os.path.join(folder_path, file.replace(".pcap", ".csv"))
            convert_pcap_to_csv(pcap_file_path, csv_file_path)
            print(f"New CSV file '{csv_file_path}' has been created:")

            df = pd.read_csv(csv_file_path)
            result_df = model.predict(df)

            columns_to_print = ["timestamp", "protocol", "src_ip", "src_port", "src_mac", "dst_ip", "dst_port", "dst_mac", "attack_type"]

            result_df = result_df[columns_to_print]

            suspicious_df = result_df[result_df["attack_type"] != "BENIGN"]
            if len(suspicious_df) != 0:
                print(suspicious_df)
                print("\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor a folder for new CSV files.")
    parser.add_argument("-t", "--test", nargs='?', const=True, help="Run a test function with an optional argument.")
    parser.add_argument("-p", "--predict", nargs='?', const=True, help="Get predictions given path to a dataset.")
    parser.add_argument("-i", "--interface", nargs='?', const=True, help="Network Interface to monitor")
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

    if not args.interface:
        print("Error: --interface arguement is required along with folder path.")
        return

    network_interface = args.interface

    try:
        check_tcp_dump()
        # Define the tcpdump command
        # tcpdump_command = f"sudo tcpdump -i {network_interface} -n -w {os.path.join(folder_path, 'capture_$(date +%Y%m%d%H%M%S).pcap')} -G 5"
        out_file = os.path.join(folder_path, "outfile-%s.pcap")
        tcpdump_command = f"sudo tcpdump -i {network_interface} -w {out_file} -G 15 -n"

        # tcpdump_process = subprocess.Popen(tcpdump_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"tcpdump process started successfully")
        time.sleep(5)
        monitor_folder(folder_path, delay=5)

    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        tcpdump_process.terminate()

if __name__ == "__main__":
    main()
