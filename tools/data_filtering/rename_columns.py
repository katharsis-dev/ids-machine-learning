from fuzzywuzzy import process

raw_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'src_mac', 'dst_mac', 'protocol', 'timestamp', 'flow_duration', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot', 'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts', 'active_max', 'active_min', 'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'cwe_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_byts']
processed_columns = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
# processed_columns = processed_columns.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

processed_columns = [value.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for value in processed_columns]
processed_columns.remove("fwd_header_length.1")

print(len(raw_columns))
print(len(processed_columns))

remove_columns = ["src_ip", "dst_ip", "src_port", "src_mac", "dst_mac", "timestamp", "protocol"]
for column in remove_columns:
    raw_columns.remove(column)


print(len(raw_columns))
print(len(processed_columns))

raw_columns.sort()
processed_columns.sort()
print(raw_columns)
print(processed_columns)

# Dictionary to store the mappings
mapping_dict = {}

# Iterate over each item in array1 and find the best match in array2
for item1 in raw_columns:
    # Use process.extractOne to find the best match
    match, score = process.extractOne(item1, processed_columns)
    
    # You can set a threshold for the similarity score to filter matches
    # if score >= 80:  # You can adjust this threshold as needed
    #     mapping_dict[item1] = match
    # else:
    # print(f"{item1} : {match}", score)
    print(f"\"{item1}\" : \"{match}\",")

result = {
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

for key in result:
    try:
        raw_columns.remove(key)
    except ValueError:
        print(key, "Failed Raw")
        exit(1)
    try:
        processed_columns.remove(result[key])
    except ValueError:
        print(result[key], "Failed Processed")
        exit(1)

print(len(raw_columns))
print(len(processed_columns))


# print(len(mapping_dict))
# # Print the mappings
# for key, value in mapping_dict.items():
#     print(f"{key} -> {value}")

# import difflib
#
# array1 = ['dst_port', 'protocol', 'flow_duration', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot', 'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts', 'active_max', 'active_min', 'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'cwe_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_byts']
#
# array2 = ['destination_port', 'flow_duration', 'total_fwd_packets', 'total_backward_packets', 'total_length_of_fwd_packets', 'total_length_of_bwd_packets', 'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean', 'bwd_packet_length_std', 'flow_bytes_s', 'flow_packets_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length', 'fwd_packets_s', 'bwd_packets_s', 'min_packet_length', 'max_packet_length', 'packet_length_mean', 'packet_length_std', 'packet_length_variance', 'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count', 'down_up_ratio', 'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size', 'fwd_header_length.1', 'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
#
# # Function to find the best match for each item in array1 within array2
# def map_items(array1, array2):
#     item_mapping = {}
#     for item1 in array1:
#         matches = difflib.get_close_matches(item1, array2)
#         if matches:
#             best_match = matches[0]
#             item_mapping[item1] = best_match
#     return item_mapping
#
# # Get the mapping
# mapping_result = map_items(array1, array2)
#
# # Print the result
# for item1, item2 in mapping_result.items():
#     print(f"{item1} -> {item2}")
#
