SAVED_MODELS_MODULE = "saved_models"

COLUMN_LENGTH_FILTERED = 77
COLUMN_LENGTH_RAW = 84

REMOVE_RAW_COLUMNS = ['src_port', 'protocol', 'timestamp', 'src_mac', 'dst_mac', 'dst_ip', 'src_ip']
BENIGN_LABEL = "benign"
SSH_LABEL = "ssh-bruteforce"
FTP_LABEL = "ftp-bruteforce"
SQL_LABEL = "sql-injection"
XSS_LABEL = "xss"
WEB_LABEL = "web-bruteforce"


FEATURE_SELECTION = [False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, True, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, True, False, False, False, False, False, True, False, False, False, False, False, False, True, True, False, True, True, True, True, True, False, False, False, False, False, True, False, False, True, False, True, False]

