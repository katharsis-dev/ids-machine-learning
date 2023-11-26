import pandas as pd
import numpy as np
import os
import sys

COLUMNS_TO_KEEP = ['ack_flag_cnt', 'active_max', 'active_mean', 'active_min', 'active_std',
       'bwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_header_len', 'bwd_iat_max',
       'bwd_iat_mean', 'bwd_iat_min', 'bwd_iat_std', 'bwd_iat_tot',
       'bwd_pkt_len_max', 'bwd_pkt_len_mean', 'bwd_pkt_len_min',
       'bwd_pkt_len_std', 'bwd_pkts_b_avg', 'bwd_pkts_s', 'bwd_psh_flags',
       'bwd_seg_size_avg', 'bwd_urg_flags', 'cwe_flag_count', 'down_up_ratio',
       'dst_port', 'ece_flag_cnt', 'fin_flag_cnt', 'flow_byts_s',
       'flow_duration', 'flow_iat_max', 'flow_iat_mean', 'flow_iat_min',
       'flow_iat_std', 'flow_pkts_s', 'fwd_act_data_pkts', 'fwd_blk_rate_avg',
       'fwd_byts_b_avg', 'fwd_header_len', 'fwd_iat_max', 'fwd_iat_mean',
       'fwd_iat_min', 'fwd_iat_std', 'fwd_iat_tot', 'fwd_pkt_len_max',
       'fwd_pkt_len_mean', 'fwd_pkt_len_min', 'fwd_pkt_len_std',
       'fwd_pkts_b_avg', 'fwd_pkts_s', 'fwd_psh_flags', 'fwd_seg_size_avg',
       'fwd_seg_size_min', 'fwd_urg_flags', 'idle_max', 'idle_mean',
       'idle_min', 'idle_std', 'init_bwd_win_byts', 'init_fwd_win_byts',
       'pkt_len_max', 'pkt_len_mean', 'pkt_len_min', 'pkt_len_std',
       'pkt_len_var', 'pkt_size_avg', 'psh_flag_cnt', 'rst_flag_cnt',
       'subflow_bwd_byts', 'subflow_bwd_pkts', 'subflow_fwd_byts',
       'subflow_fwd_pkts', 'syn_flag_cnt', 'tot_bwd_pkts', 'tot_fwd_pkts',
       'totlen_bwd_pkts', 'totlen_fwd_pkts', 'urg_flag_cnt', 'label']


def process_csv(file_path, output_directory):

    filtered_file_name = os.path.join(output_directory, "filter_" + os.path.basename(file_path))

    os.makedirs(output_directory, exist_ok=True)
    # For demonstration, let's just print the content of each file
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("/", "_")

    df = df[COLUMNS_TO_KEEP]
    df = df.dropna()
    df = df.drop_duplicates(keep="first")
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]
    df.to_csv(filtered_file_name, index=False)


def main():
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        print("Missing Arguement")
        exit(1)

    # Define the output directory
    output_directory = os.path.join(target_directory, 'filter')

    # List all files in the current working directory
    file_list = os.listdir(target_directory)

    # Filter CSV files
    csv_files = [file for file in file_list if file.endswith('.csv')]

    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(target_directory, csv_file)
        process_csv(file_path, output_directory)

if __name__ == "__main__":
    main()

