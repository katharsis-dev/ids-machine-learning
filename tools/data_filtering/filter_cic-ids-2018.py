import pandas as pd
import numpy as np
import os
import sys

def invert_dict(dictionary):
    result = {}
    for key in dictionary:
        result[dictionary[key]] = key
    return result

RAW_TO_FILTERED = {
    "ack_flag_cnt" : "ack_flag_count",
    "active_max" : "active_max",
    "active_mean" : "active_mean",
    "active_min" : "active_min",
    "active_std" : "active_std",
    "bwd_blk_rate_avg" : "bwd_avg_bulk_rate",
    "bwd_byts_b_avg" : "bwd_avg_bytes_bulk",
    "bwd_header_len" : "bwd_header_length",
    "bwd_iat_max" : "bwd_iat_max",
    "bwd_iat_mean" : "bwd_iat_mean",
    "bwd_iat_min" : "bwd_iat_min",
    "bwd_iat_std" : "bwd_iat_std",
    "bwd_iat_tot" : "bwd_iat_total",
    "bwd_pkt_len_max" : "bwd_packet_length_max",
    "bwd_pkt_len_mean" : "bwd_packet_length_mean",
    "bwd_pkt_len_min" : "bwd_packet_length_min",
    "bwd_pkt_len_std" : "bwd_packet_length_std",
    "bwd_pkts_b_avg" : "bwd_avg_packets_bulk",
    "bwd_pkts_s" : "bwd_packets_s",
    "bwd_psh_flags" : "bwd_psh_flags",
    "bwd_seg_size_avg" : "avg_bwd_segment_size",
    "bwd_urg_flags" : "bwd_urg_flags",
    "cwe_flag_count" : "cwe_flag_count",
    "down_up_ratio" : "down_up_ratio",
    "dst_port" : "destination_port",
    "ece_flag_cnt" : "ece_flag_count",
    "fin_flag_cnt" : "fin_flag_count",
    "flow_byts_s" : "flow_bytes_s",
    "flow_duration" : "flow_duration",
    "flow_iat_max" : "flow_iat_max",
    "flow_iat_mean" : "flow_iat_mean",
    "flow_iat_min" : "flow_iat_min",
    "flow_iat_std" : "flow_iat_std",
    "flow_pkts_s" : "flow_packets_s",
    "fwd_act_data_pkts" : "act_data_pkt_fwd",
    "fwd_blk_rate_avg" : "fwd_avg_bulk_rate",
    "fwd_byts_b_avg" : "fwd_avg_bytes_bulk",
    "fwd_header_len" : "fwd_header_length",
    "fwd_iat_max" : "fwd_iat_max",
    "fwd_iat_mean" : "fwd_iat_mean",
    "fwd_iat_min" : "fwd_iat_min",
    "fwd_iat_std" : "fwd_iat_std",
    "fwd_iat_tot" : "fwd_iat_total",
    "fwd_pkt_len_max" : "fwd_packet_length_max",
    "fwd_pkt_len_mean" : "fwd_packet_length_mean",
    "fwd_pkt_len_min" : "fwd_packet_length_min",
    "fwd_pkt_len_std" : "fwd_packet_length_std",
    "fwd_pkts_b_avg" : "fwd_avg_packets_bulk",
    "fwd_pkts_s" : "fwd_packets_s",
    "fwd_psh_flags" : "fwd_psh_flags",
    "fwd_seg_size_avg" : "avg_fwd_segment_size",
    "fwd_seg_size_min" : "min_seg_size_forward",
    "fwd_urg_flags" : "fwd_urg_flags",
    "idle_max" : "idle_max",
    "idle_mean" : "idle_mean",
    "idle_min" : "idle_min",
    "idle_std" : "idle_std",
    "init_bwd_win_byts" : "init_win_bytes_backward",
    "init_fwd_win_byts" : "init_win_bytes_forward",
    "pkt_len_max" : "min_packet_length",
    "pkt_len_mean" : "packet_length_mean",
    "pkt_len_min" : "max_packet_length",
    "pkt_len_std" : "packet_length_std",
    "pkt_len_var" : "packet_length_variance",
    "pkt_size_avg" : "average_packet_size",
    "psh_flag_cnt" : "psh_flag_count",
    "rst_flag_cnt" : "rst_flag_count",
    "subflow_bwd_byts" : "subflow_bwd_bytes",
    "subflow_bwd_pkts" : "subflow_bwd_packets",
    "subflow_fwd_byts" : "subflow_fwd_bytes",
    "subflow_fwd_pkts" : "subflow_fwd_packets",
    "syn_flag_cnt" : "syn_flag_count",
    "tot_bwd_pkts" : "total_backward_packets",
    "tot_fwd_pkts" : "total_fwd_packets",
    "totlen_bwd_pkts" : "total_length_of_bwd_packets",
    "totlen_fwd_pkts" : "total_length_of_fwd_packets",
    "urg_flag_cnt" : "urg_flag_count"
}

FILTERED_TO_RAW = invert_dict(RAW_TO_FILTERED)

def process_csv(file_path, output_directory):

    filtered_file_name = os.path.join(output_directory, "filter_" + os.path.basename(file_path))

    os.makedirs(output_directory, exist_ok=True)
    # For demonstration, let's just print the content of each file
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("/", "_")

    columns_to_drop = ['fwd_header_length.1', "protocol", "timestamp"]
    for column in columns_to_drop:
        if column in df.columns:
            df = df.drop(column, axis=1)

    # print(df.columns, len(df.columns))
    # print(list(RAW_TO_FILTERED.keys()), len(list(RAW_TO_FILTERED.keys())))
    # exit()

    raw_columns = list(RAW_TO_FILTERED.keys())
    for column in df.columns:
        if column != "label":
            raw_columns.remove(column)

    if len(raw_columns) != 0:
        print(raw_columns)
        raise ValueError("Column Names Do Not Match")

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

