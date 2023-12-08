import numpy as np
import pandas as pd
import os 
from tqdm import tqdm

# convert data types to reduce memory usage
dict_types = {
          'flow_duration': np.float32, 
          'Header_Length': np.int32, 
          'Protocol Type': np.float32, 
          'Duration': np.float32, 
          'Rate': np.uint32, 
          'Srate': np.uint32, 
          'Drate': np.float32, 
          'fin_flag_number': np.uint8, 
          'syn_flag_number': np.uint8, 
          'rst_flag_number': np.uint8, 
          'psh_flag_number': np.uint8, 
          'ack_flag_number': np.uint8, 
          'ece_flag_number': np.uint8, 
          'cwr_flag_number': np.uint8, 
          'ack_count': np.float16, 
          'syn_count': np.float16, 
          'fin_count': np.uint16, 
          'urg_count': np.uint16, 
          'rst_count': np.uint16, 
          'HTTP': np.uint8, 
          'HTTPS': np.uint8, 
          'DNS': np.uint8, 
          'Telnet': np.uint8, 
          'SMTP': np.uint8, 
          'SSH': np.uint8, 
          'IRC': np.uint8, 
          'TCP': np.uint8, 
          'UDP': np.uint8,
          'DHCP': np.uint8, 
          'ARP': np.uint8, 
          'ICMP': np.uint8, 
          'IPv': np.uint8, 
          'LLC': np.uint8, 
          'Tot sum': np.float32, 
          'Min': np.float32, 
          'Max': np.float32, 
          'AVG': np.float32, 
          'Std': np.float32, 
          'Tot size': np.float32, 
          'IAT': np.float32, 
          'Number': np.float32, 
          'Magnitue': np.float32, 
          'Radius': np.float32, 
          'Covariance': np.float32, 
          'Variance': np.float32, 
          'Weight': np.float32,
          'label': np.uint8
          }

def convert_type(
    df: pd.DataFrame
    ) -> pd.DataFrame: 
    """
    convert data type yo reduce memory usage
    """
    # convert type
    for col, type in dict_types.items():
        df[col] = df[col].astype(type)

    # format column
    # df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def create_merged_file(
    directory: str,
    is_convert_types: bool=True,
    file_type: str='csv' 
)-> None:
    """
    create merged file from many files replaced in directory

    directory:  directory of dataset
    is_convert_types: if True - call function convert_type to convert data types to reduce memory usage and map attacks
    file_type: 'csv' or 'pkl'
    """
    list_dir = os.listdir(directory)
    # read and load all csv files into single list
    df_merged = []
    for csv in tqdm(list_dir):
        df_temp = pd.read_csv(directory +"\\" + csv)
        if is_convert_types:
            df_temp = convert_type(df_temp)
            df_merged.append(df_temp)
        else:
            df_merged.append(df_temp)

    # hand over created list in single dataframe
    merged_df = pd.concat(df_merged, ignore_index=True)

    # write created dataframe into file
    if file_type == 'csv':
    # csv
        out_file_name = directory + '\\' + 'merged.csv'
        merged_df.to_csv(out_file_name, index=False)
        print(f'Merged is complited.\n{out_file_name} is created')
    # pkl
    elif file_type == 'pkl':
        out_file_name = directory + '\\' + 'merged.pkl'
        merged_df.to_pickle(directory + '\\' + 'merged.pkl')
        print(f'Merged is complited.\n{out_file_name} is created')
    else:
        print('File is not created')



# binary classification: divide into 2 classes (attacks and Benign)
dict_binary = {}
# Attacks
# DDoS
dict_binary['DDoS-RSTFINFlood'] = 'Attack'
dict_binary['DDoS-ICMP_Flood'] = 'Attack'
dict_binary['DDoS-HTTP_Flood'] = 'Attack'
dict_binary['DDoS-SynonymousIP_Flood'] = 'Attack'
dict_binary['DDoS-SYN_Flood'] = 'Attack'
dict_binary['DDoS-PSHACK_Flood'] = 'Attack'
dict_binary['DDoS-UDP_Flood'] = 'Attack'
dict_binary['DDoS-TCP_Flood'] = 'Attack'
dict_binary['DDoS-ACK_Fragmentation'] = 'Attack'
dict_binary['DDoS-ICMP_Fragmentation'] = 'Attack'
dict_binary['DDoS-UDP_Fragmentation'] = 'Attack'
dict_binary['DDoS-SlowLoris'] = 'Attack'
# DoS
dict_binary['DoS-TCP_Flood'] = 'Attack'
dict_binary['DoS-UDP_Flood'] = 'Attack'
dict_binary['DoS-SYN_Flood'] = 'Attack'
dict_binary['DoS-HTTP_Flood'] = 'Attack'
# Mirai
dict_binary['Mirai-greeth_flood'] = 'Attack'
dict_binary['Mirai-udpplain'] = 'Attack'
dict_binary['Mirai-greip_flood'] = 'Attack'
# Recon
dict_binary['Recon-PortScan'] = 'Attack'
dict_binary['Recon-OSScan'] = 'Attack'
dict_binary['Recon-HostDiscovery'] = 'Attack'
dict_binary['Recon-PingSweep'] = 'Attack'
dict_binary['VulnerabilityScan'] = 'Attack'
# Spoofing
dict_binary['DNS_Spoofing'] = 'Attack'
dict_binary['MITM-ArpSpoofing'] = 'Attack'
# Web
dict_binary['BenignTraffic'] = 'Attack'
dict_binary['XSS'] = 'Attack'
dict_binary['CommandInjection'] = 'Attack'
dict_binary['Backdoor_Malware'] = 'Attack'
dict_binary['BrowserHijacking'] = 'Attack'
dict_binary['SqlInjection'] = 'Attack'
dict_binary['Uploading_Attack'] = 'Attack'
# BrueForce
dict_binary['DictionaryBruteForce'] = 'Attack'
# Benign
dict_binary['BenignTraffic'] = 'BenignTraffic'


# multiclass classification: divide into 8 classes (7 attacks and 1 Benign)
dict_group = {}
# DDoS
dict_group['DDoS-RSTFINFlood'] = 'DDos'
dict_group['DDoS-ICMP_Flood'] = 'DDos'
dict_group['DDoS-HTTP_Flood'] = 'DDos'
dict_group['DDoS-SynonymousIP_Flood'] = 'DDos'
dict_group['DDoS-SYN_Flood'] = 'DDos'
dict_group['DDoS-PSHACK_Flood'] = 'DDos'
dict_group['DDoS-UDP_Flood'] = 'DDos'
dict_group['DDoS-TCP_Flood'] = 'DDos'
dict_group['DDoS-ACK_Fragmentation'] = 'DDos'
dict_group['DDoS-ICMP_Fragmentation'] = 'DDos'
dict_group['DDoS-UDP_Fragmentation'] = 'DDos'
dict_group['DDoS-SlowLoris'] = 'DDos'
# DoS
dict_group['DoS-TCP_Flood'] = 'Dos'
dict_group['DoS-UDP_Flood'] = 'Dos'
dict_group['DoS-SYN_Flood'] = 'Dos'
dict_group['DoS-HTTP_Flood'] = 'Dos'
# Mirai
dict_group['Mirai-greeth_flood'] = 'Mirai'
dict_group['Mirai-udpplain'] = 'Mirai'
dict_group['Mirai-greip_flood'] = 'Mirai'
# Recon
dict_group['Recon-PortScan'] = 'Recon'
dict_group['Recon-OSScan'] = 'Recon'
dict_group['Recon-HostDiscovery'] = 'Recon'
dict_group['Recon-PingSweep'] = 'Recon'
dict_group['VulnerabilityScan'] = 'Recon'
# Spoofing
dict_group['DNS_Spoofing'] = 'Spoofing'
dict_group['MITM-ArpSpoofing'] = 'Spoofing'
# Web
dict_group['BenignTraffic'] = 'Web'
dict_group['XSS'] = 'Web'
dict_group['CommandInjection'] = 'Web'
dict_group['Backdoor_Malware'] = 'Web'
dict_group['BrowserHijacking'] = 'Web'
dict_group['SqlInjection'] = 'Web'
dict_group['Uploading_Attack'] = 'Web'
# BrueForce
dict_group['DictionaryBruteForce'] = 'BrueForce'
# Benign
dict_group['BenignTraffic'] = 'BenignTraffic'

def target_renaiming(
    df: pd.DataFrame,
    classification_type: str,
    is_write_file: bool =True,
    directory: str = None,
    output_file: str ='new_file',
    file_type: str='csv' 
    ) -> pd.DataFrame:
    """
    create new dataset or/and file with new data types and new target depends of classification type
    work in place
    _____________
    output_file: full name of csv file with path 
    classification_type: binary, group, individual
    is_write_file: if True -create new file
    directory: directory name to create file
    output_file: name of new file
    file_type: 'csv' or 'pkl'
    """
    # df = pd.read_csv(input_file)
    # df = convert_type(df)

    if classification_type == 'binary':
        # map target column to the dict_binary
        df['label'] = df['label'].map(dict_binary)
        output_file = directory + output_file +'_binary.' + file_type
    elif classification_type == 'group':
        # map target column to the dict_group
        df['label'] = df['label'].map(dict_group)
        output_file = directory + output_file +'_group.' + file_type
    elif classification_type == 'individual':
        # no map 
        output_file = directory + output_file +'_individual.' + file_type

    # write new dataset in file
    if is_write_file:
        if file_type == 'csv':
        # csv
            df.to_csv(output_file, index=False)
            print(output_file +' is created')
        # pkl
        elif file_type == 'pkl':
            df.to_pickle(output_file, index=False)
            print(output_file +' is created')
        else:
            print('File is not created')
    else:
        print('dataset is created\nFile is not created')
    return df



# Map IANA Protocol numbers to strings
dic_iana_protocol = { 
    "0": "HOPOPT", "1": "ICMP", "2": "IGMP", "3": "GGP", "4": "IPv4", "5": "ST", 
    "6": "TCP", "7": "CBT", "8": "EGP", "9": "IGP", "10": "BBN-RCC-MON", "11": "NVP-II", 
    "12": "PUP", "13": "ARGUS (deprecated)", "14": "EMCON", "15": "XNET", "16": "CHAOS", 
    "17": "UDP", "18": "MUX", "19": "DCN-MEAS", "20": "HMP", "21": "PRM", "22": "XNS-IDP", 
    "23": "TRUNK-1", "24": "TRUNK-2", "25": "LEAF-1", "26": "LEAF-2", "27": "RDP", 
    "28": "IRTP", "29": "ISO-TP4", "30": "NETBLT", "31": "MFE-NSP", "32": "MERIT-INP", 
    "33": "DCCP", "34": "3PC", "35": "IDPR", "36": "XTP", "37": "DDP", "38": "IDPR-CMTP", 
    "39": "TP++", "40": "IL", "41": "IPv6", "42": "SDRP", "43": "IPv6-Route", 
    "44": "IPv6-Frag", "45": "IDRP", "46": "RSVP", "47": "GRE", "48": "DSR", "49": "BNA", 
    "50": "ESP", "51": "AH", "52": "I-NLSP", "53": "SWIPE (deprecated)", "54": "NARP", 
    "55": "MOBILE", "56": "TLSP", "57": "SKIP", "58": "IPv6-ICMP", "59": "IPv6-NoNxt", 
    "60": "IPv6-Opts", "62": "CFTP", "64": "SAT-EXPAK", "65": "KRYPTOLAN", "66": "RVD", 
    "67": "IPPC", "69": "SAT-MON", "70": "VISA", "71": "IPCV", "72": "CPNX", "73": "CPHB", 
    "74": "WSN", "75": "PVP", "76": "BR-SAT-MON", "77": "SUN-ND", "78": "WB-MON", 
    "79": "WB-EXPAK", "80": "ISO-IP", "81": "VMTP", "82": "SECURE-VMTP", "83": "VINES", 
    "84": "IPTM", "85": "NSFNET-IGP", "86": "DGP", "87": "TCF", "88": "EIGRP", 
    "89": "OSPFIGP", "90": "Sprite-RPC", "91": "LARP", "92": "MTP", "93": "AX.25", 
    "94": "IPIP", "95": "MICP (deprecated)","96": "SCC-SP", "97": "ETHERIP", "98": "ENCAP", 
    "100": "GMTP", "101": "IFMP", "102": "PNNI", "103": "PIM", "104": "ARIS", "105": "SCPS", 
    "106": "QNX", "107": "A/N", "108": "IPComp", "109": "SNP", "110": "Compaq-Peer", 
    "111": "IPX-in-IP", "112": "VRRP", "113": "PGM", "114": "", "115": "L2TP", "116": "DDX",  
    "117": "IATP", "118": "STP", "119": "SRP", "120": "UTI", "121": "SMP", 
    "122": "SM (deprecated)", "123": "PTP","124": "ISIS over IPv4", "125": "FIRE", 
    "126": "CRTP", "127": "CRUDP", "128": "SSCOPMCE", "129": "IPLT", "130": "SPS", 
    "131": "PIPE", "132": "SCTP",  "133": "FC", "134": "RSVP-E2E-IGNORE", 
    "135": "Mobility Header", "136": "UDPLite", "137": "MPLS-in-IP", "138": "manet", 
    "139": "HIP", "140": "Shim6", "141": "WESP", "142": "ROHC", "143": "Ethernet", 
    "144": "AGGFRAG", "145": "NSH"
}
def iana_convert(
    df: pd.DataFrame
    ) -> pd.DataFrame: 
    """
    convert 'Protocol Type' to string
    """
    df["Protocol Type"] = df["Protocol Type"].apply(lambda num : dic_iana_protocol[ str(int(num)) ])
    return df

# Creating a dictionary of attack types for 33 attack classes + 1 for benign traffic
dict_34_classes = {'BenignTraffic': 0 ,                                                                                                                         # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':2,  'DDoS-SYN_Flood':3, 'DDoS-UDP_Flood':4, 'DDoS-TCP_Flood':5, 
                   'DDoS-ICMP_Flood':6, 'DDoS-SynonymousIP_Flood':7, 'DDoS-ACK_Fragmentation':8, 'DDoS-UDP_Fragmentation':9, 'DDoS-ICMP_Fragmentation':10, 
                   'DDoS-SlowLoris':11, 'DDoS-HTTP_Flood':12,                                                                                                   # DDoS
                   'DoS-UDP_Flood':13, 'DoS-SYN_Flood':14, 'DoS-TCP_Flood':15, 'DoS-HTTP_Flood':16,                                                             # DoS
                   'Mirai-greeth_flood': 17, 'Mirai-greip_flood': 18, 'Mirai-udpplain': 19,                                                                     # Mirai 
                   'Recon-PingSweep': 20, 'Recon-OSScan': 21, 'Recon-PortScan': 22, 'VulnerabilityScan': 23, 'Recon-HostDiscovery': 24,                         # Reconnaissance
                   'DNS_Spoofing': 25, 'MITM-ArpSpoofing': 26,                                                                                                  # Spoofing
                   'BrowserHijacking': 27, 'Backdoor_Malware': 28, 'XSS': 29, 'Uploading_Attack': 30, 'SqlInjection': 31, 'CommandInjection': 32,               # Web
                   'DictionaryBruteForce': 33}                                                                                                                  # Brute Force 

dict_8_classes = {'BenignTraffic': 0 ,                                                                                                                          # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':1,  'DDoS-SYN_Flood':1, 'DDoS-UDP_Flood':1, 'DDoS-TCP_Flood':1, 
                   'DDoS-ICMP_Flood':1, 'DDoS-SynonymousIP_Flood':1, 'DDoS-ACK_Fragmentation':1, 'DDoS-UDP_Fragmentation':1, 'DDoS-ICMP_Fragmentation':1, 
                   'DDoS-SlowLoris':1, 'DDoS-HTTP_Flood':1,                                                                                                     # DDoS
                   'DoS-UDP_Flood':2, 'DoS-SYN_Flood':2, 'DoS-TCP_Flood':2, 'DoS-HTTP_Flood':2,                                                                 # DoS
                   'Mirai-greeth_flood': 3, 'Mirai-greip_flood': 3, 'Mirai-udpplain': 3,                                                                        # Mirai 
                   'Recon-PingSweep': 4, 'Recon-OSScan': 4, 'Recon-PortScan': 4, 'VulnerabilityScan': 4, 'Recon-HostDiscovery': 4,                              # Reconnaissance
                   'DNS_Spoofing': 5, 'MITM-ArpSpoofing': 5,                                                                                                    # Spoofing
                   'BrowserHijacking': 6, 'Backdoor_Malware': 6, 'XSS': 6, 'Uploading_Attack': 6, 'SqlInjection': 6, 'CommandInjection': 6,                     # Web
                   'DictionaryBruteForce': 7}                                                                                                                                   # 7 - Brute Force

dict_2_classes = {'BenignTraffic': 0 ,                                                                                                                          # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':1,  'DDoS-SYN_Flood':1, 'DDoS-UDP_Flood':1, 'DDoS-TCP_Flood':1, 
                   'DDoS-ICMP_Flood':1, 'DDoS-SynonymousIP_Flood':1, 'DDoS-ACK_Fragmentation':1, 'DDoS-UDP_Fragmentation':1, 'DDoS-ICMP_Fragmentation':1, 
                   'DDoS-SlowLoris':1, 'DDoS-HTTP_Flood':1,                                                                                                     # DDoS
                   'DoS-UDP_Flood':1, 'DoS-SYN_Flood':1, 'DoS-TCP_Flood':1, 'DoS-HTTP_Flood':1,                                                                 # DoS
                   'Mirai-greeth_flood': 1, 'Mirai-greip_flood': 1, 'Mirai-udpplain': 1,                                                                        # Mirai 
                   'Recon-PingSweep': 1, 'Recon-OSScan': 1, 'Recon-PortScan': 1, 'VulnerabilityScan': 1, 'Recon-HostDiscovery': 1,                              # Reconnaissance
                   'DNS_Spoofing': 1, 'MITM-ArpSpoofing': 1,                                                                                                    # Spoofing
                   'BrowserHijacking': 1, 'Backdoor_Malware': 1, 'XSS': 1, 'Uploading_Attack': 1, 'SqlInjection': 1, 'CommandInjection': 1,                     # Web
                   'DictionaryBruteForce': 1} 

def convert_to_gigital_target(
        df: pd.DataFrame,
        classification_type: str,
)-> pd.DataFrame:
    """
    convert label from object or string to digital

    classification_type: binary, group, individual
    """

    if classification_type == 'binary':
        df['label'] = df['label'].map(dict_2_classes)
    elif classification_type == 'group':
        df['label'] = df['label'].map(dict_8_classes)
    elif classification_type == 'individual':
        df['label'] = df['label'].map(dict_34_classes)
    return df


# TO DO
def take_data_part():
    """
    take part of all data
    """
    pass
