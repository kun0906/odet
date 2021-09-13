"""Extract feat sets from pcaps and store the results as normal.csv and abnormal.csv seperately
    data: pcap
    feat sets: iat, size, iat_size, fft_iat, ..., samp_num, samp_size, ...

Note:
    avoiding doing everything
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE
# 1. put _config as the first line
try:
    # import reprst._config_kjl
    from iot23._config_iot23 import *
except Exception as e:
    print(f'no _config_kjl.py exists, so load the default one')
    from itod._config import *

# 2. standard libraries
import argparse
import time

# 3. third-party packages
import numpy as np
from sklearn.utils import shuffle

# 4. your own package
from itod.data.pcap import pcap2flows, label_flows, get_header_features, _flows_to_iats_sizes, \
    _flows_to_samps, _flows_to_stats, flows2subflows, choose_flows
from itod.utils.utils import dump_data, data_info, unpickle_data
from itod_kjl.data.pcap.IoT23 import label_flows_IoT23

# for debug
FEAT_SETS = ['iat_size']


# all feat sets for each data
# FEAT_SETS = ['iat', 'size', 'iat_size', 'fft_iat', 'fft_size', 'fft_iat_size',
#              'stat', 'samp_num', 'samp_size', 'samp_num_size', 'fft_samp_num',
#              'fft_samp_size', 'fft_samp_num_size']


def _get_fft(features, fft_bin='', fft_part='real'):
    """Do fft transform of features

    Parameters
    ----------
    features: features

    fft_bin: int
        the dimension of transformed features
    fft_part: str
        'real' or 'real+imaginary' transformation

    feat_set: str

    Returns
    -------
    fft_features:
        transformed fft features
    """
    if fft_part == 'real':  # default
        fft_features = [np.real(np.fft.fft(v, n=fft_bin)) for v in features]

    elif fft_part == 'real+imaginary':
        fft_features = []
        for i, v in enumerate(features):
            complex_v = np.fft.fft(v, fft_bin)
            if i == 0:
                print(f'dimension of the real part: {len(np.real(complex_v))}, '
                      f'dimension of the imaginary part: {len(np.imag(complex_v))}')
            v = np.concatenate([np.real(complex_v), np.imag(complex_v)], axis=np.newaxis)
            fft_features.append(v)

    else:
        print(f'fft_part: {fft_part} is not correct, please modify it and retry')
        return -1

    return np.asarray(fft_features, dtype=float)


def _fix_data(features, dim):
    """Fix data by appending '0' or cutting off

    Parameters
    ----------
    features

    dim: int
        the fixed dimension of features

    Returns
    -------
    fixed_features:
        the fixed features
    """
    fixed_features = []
    for feat in features:
        feat = list(feat)
        if len(feat) > dim:
            feat = feat[:dim]
        else:
            feat += [0] * (dim - len(feat))

        fixed_features.append(np.asarray(feat, dtype=float))

    return np.asarray(fixed_features, dtype=float)


def _extract_header(flows):
    # convert Unix timestamp arrival times into interpacket intervals
    flows = [(fid, np.diff(times), pkts) for (fid, times, pkts) in flows]  # No need to use sizes[1:]
    data_header = []
    for fid, times, pkts in flows:  # fid, IAT, pkt_len
        data_header.append((fid, get_header_features(pkts)))  # (fid, np.array())

    return data_header


def extract_features_intf(flows, feat_set, q_feat=0.9, q_samp=None, dim=None, dim_header=None, header=False):
    try:
        if header:
            data_header = _extract_header(flows)
        else:
            data_header = {}

        if feat_set in ['iat', 'size', 'iat_size']:
            data_feat = _flows_to_iats_sizes(flows, feat_set, verbose=True)
        elif feat_set in ['stats']:
            data_feat = _flows_to_stats(flows)
        elif 'samp' in feat_set:
            durations = [np.max(pkt_times) - np.min(pkt_times) for fids, pkt_times, pkts in flows]
            dim_iat = int(np.quantile(durations, q=q_feat))
            samp_rate = np.quantile([v / dim_iat for v in durations], q=q_samp)
            data_feat = _flows_to_samps(flows, sampling_type='rate', sampling=samp_rate,
                                        sampling_feature=feat_set, verbose=True)

    except Exception as e:
        msg = f'Error: {e}'
        le(msg)
        raise ValueError(msg)

    # data_feat, _ = _extract_features(flows, feat_set, header)
    # fix features
    if not dim:  # dim != None
        dim = int(np.quantile([len(feat) for fid, feat in data_feat], q_feat))

    data_feat = [v for fid, v in data_feat]

    data_feat = _fix_data(data_feat, dim)
    fft_data_feat = _get_fft(data_feat, fft_bin=dim)

    if header:
        # fix features
        if not dim_header:  # dim != None
            dim_header = int(np.quantile([len(feat) for fids, feat in data_header], q_feat))

        data_header = [v for fid, v in data_header]
        data_header = _fix_data(data_header, dim_header)
        fft_data_header = _get_fft(data_header, dim_header)

        data_feat = np.hstack([data_header, data_feat])
        fft_data_feat = np.hstack([fft_data_header, fft_data_feat])
    else:
        dim_header = None

    return data_feat, fft_data_feat, dim, dim_header


def get_each_intf(flows_normal, flows_abnormal, feat_set='iat', q_feat=0.9, q_samp=None, header=False,
                  prefix_pth="pth.join(dir_out, data_name)"):
    key_pth = pth.join(prefix_pth, feat_set, f'header:{header}')
    key_fft_pth = pth.join(prefix_pth, f'fft_{feat_set}', f'header:{header}')
    if not pth.exists(key_pth): os.makedirs(key_pth)
    if not pth.exists(key_fft_pth): os.makedirs(key_fft_pth)

    try:
        # 1. get normal csv data
        feat_normal, fft_feat_normal, dim_normal, dim_normal_header = extract_features_intf(flows_normal, feat_set,
                                                                                            q_feat, q_samp, dim=None,
                                                                                            dim_header=None,
                                                                                            header=header)
        lg(f'feat_normal.shape: {feat_normal.shape}, fft_feat_normal.shape: {fft_feat_normal.shape}')
        # 2. get abnormal csv data
        feat_abnormal, fft_feat_abnormal, _, _ = extract_features_intf(flows_abnormal, feat_set, q_feat, q_samp,
                                                                       dim=dim_normal, dim_header=dim_normal_header,
                                                                       header=header)
        lg(f'feat_abnormal.shape: {feat_abnormal.shape}, fft_feat_abnormal.shape: {fft_feat_abnormal.shape}')
    except Exception as e:
        msg = f'{extract_features_intf.__name__}, error:{e}'
        le(msg)
        raise ValueError(msg)

    # simple is better than complex
    if 'samp' in feat_set:
        # save normal
        pth_feat_normal = pth.join(key_pth, str(q_samp), f'normal.csv')
        # save fft normal
        pth_fft_normal = pth.join(key_fft_pth, str(q_samp), 'normal.csv')

        # save abnormal
        pth_feat_abnormal = pth.join(key_pth, str(q_samp), 'abnormal.csv')
        # save fft abnormal
        pth_fft_abnormal = pth.join(key_fft_pth, str(q_samp), 'abnormal.csv')

    else:
        # save normal
        pth_feat_normal = pth.join(key_pth, 'normal.csv')
        # save fft normal
        pth_fft_normal = pth.join(key_fft_pth, 'normal.csv')

        # save abnormal
        pth_feat_abnormal = pth.join(key_pth, 'abnormal.csv')
        # save fft abnormal
        pth_fft_abnormal = pth.join(key_fft_pth, 'abnormal.csv')

    if not pth.exists(pth.dirname(pth_feat_normal)): os.makedirs(pth.dirname(pth_feat_normal))
    features2csv(feat_normal, file_out=pth_feat_normal)  # save features to csv

    if not pth.exists(pth.dirname(pth_fft_normal)): os.makedirs(pth.dirname(pth_fft_normal))
    features2csv(fft_feat_normal, file_out=pth_fft_normal)  # save features to csv

    if not pth.exists(pth.dirname(pth_feat_abnormal)): os.makedirs(pth.dirname(pth_feat_abnormal))
    features2csv(feat_abnormal, file_out=pth_feat_abnormal)  # save features to csv

    if not pth.exists(pth.dirname(pth_fft_abnormal)): os.makedirs(pth.dirname(pth_fft_abnormal))
    features2csv(fft_feat_abnormal, file_out=pth_fft_abnormal)  # save features to csv

    result = {'normal': (feat_normal, pth_feat_normal),
              'abnormal': (feat_abnormal, pth_feat_abnormal),
              'header': header, 'q_feat': q_feat}
    fft_result = {'normal': (fft_feat_normal, pth_fft_normal),
                  'abnormal': (fft_feat_abnormal, pth_fft_abnormal),
                  'header': header, 'q_feat': q_feat}
    lg(result)
    lg(fft_result)

    return result, fft_result


def features2csv(feat_arr, file_out=''):
    np.savetxt(file_out, [p for p in feat_arr], delimiter=',', fmt='%s')


def _get_path(dir_in, data_name):
    if data_name == 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5':
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8':
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9':
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14':
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15':
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.15_AGMT.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.15_AGMT.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'pc_10.42.0.1_normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name, 'pc_10.42.0.119_anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'MIRAI/benign-dec-EZVIZ-ip_src-192.168.0.13-normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                'MIRAI/mirai-udpflooding-1-dec-EZVIZ-ip_src-192.168.0.13-anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, '201912071400-10000000pkts_00000_src_202_171_168_50_normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                '201912071400-10000000pkts_00000_src_202_4_27_109_anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'samsung_camera-2daysactiv-src_192.168.143.42-normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                'samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'samsung_fridge-2daysactiv-src_192.168.143.43-normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                'ssamsung_fridge-2daysactiv-src_192.168.143.43-anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48':
        # normal and abormal are independent
        pth_normal = pth.join(dir_in, data_name, 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap')
        pth_abnormal = pth.join(dir_in, data_name,
                                'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap')
        pth_labels_normal, pth_labels_abnormal = None, None

    elif data_name == 'CTU-IoT-Malware-Capture-20-1':
        ip = '192.168.100.103'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None


    elif data_name == 'CTU-IoT-Malware-Capture-8-1':
        ip = '192.168.100.113'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None


    elif data_name == 'CTU-IoT-Malware-Capture-7-1':
        ip = '192.168.100.108'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'CTU-IoT-Malware-Capture-3-1':
        ip = '192.168.2.5'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'CTU-IoT-Malware-Capture-1-1':
        ip = '192.168.100.103'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'CTU-IoT-Malware-Capture-33-1':
        ip = '192.168.100.103'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    elif data_name == 'CTU-IoT-Malware-Capture-48-1':
        ip = '192.168.1.200'
        # normal and abormal are mixed together
        pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
        pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')

        pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

    # elif data_name == 'CTU-IoT-Malware-Capture-9-1':
    #     ip = '192.168.1.200'
    #     # normal and abormal are mixed together
    #     pth_pcap_mixed = pth.join(dir_in, data_name, f'src_{ip}.pcap')
    #     pth_labels_mixed = pth.join(dir_in, data_name, f'src_{ip}.csv')
    #
    #     pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
    #
    #
    # elif data_name == 'CTU-IoT-Malware-Capture-5-1':
    #     # normal and abormal are independent
    #     pth_normal = pth.join(dir_in, data_name, '2018-09-21-capture.pcap')
    #     pth_abnormal = pth.join(dir_in, data_name,
    #                             'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
    #     pth_labels_normal, pth_labels_abnormal = None, None

    else:
        msg = f'{data_name} does not exist'
        le(msg)
        raise ValueError(msg)

    lg(pth_normal)
    lg(pth_abnormal)
    lg(pth_labels_normal)
    lg(pth_labels_abnormal)

    return pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal


def get_subflows(dir_in, data_name, data_type='mix', q_flow=0.9):
    if data_type == 'mix':
        # normal and abnormal packets are mixed into one pcap
        # 1) get flows mixed
        pcap_mixed, _, pth_labels_mixed, _ = _get_path(dir_in, data_name)
        lg(pcap_mixed)
        lg(pth_labels_mixed)
        flows_mixed, num_pkts = pcap2flows(pcap_mixed)

        # 2) use labels to seperate normal and abnormal flows
        fids, labels = label_flows_IoT23(flows_mixed, pth_label=pth_labels_mixed)
        flows_normal = [(fid_flow, pkt_times, pkts) for (fid_flow, pkt_times, pkts), (fid, label) in
                        zip(flows_mixed, zip(fids, labels)) if fid_flow == fid and label == "normal".upper()]
        flows_abnormal = [(fid_flow, pkt_times, pkts) for (fid_flow, pkt_times, pkts), (fid, label) in
                          zip(flows_mixed, zip(fids, labels)) if fid_flow == fid and label == "abnormal".upper()]
    else:
        # 1) get full flows: normal and abnormal flows
        pcap_normal, pcap_abnormal, pth_label_normal, pth_label_abnormal = _get_path(dir_in, data_name)
        flows_normal, num_pkts = pcap2flows(pcap_normal)
        flows_abnormal, num_pkts = pcap2flows(pcap_abnormal)

        # 2) no need to separate

    # dump all flows, not subflows
    pth_flows_normal = pth.join(dir_in, data_name, 'all-flows-normal.dat')
    pth_flows_abnormal = pth.join(dir_in, data_name, 'all-flows-abnormal.dat')
    if not pth.exists(pth.dirname(pth_flows_normal)): os.makedirs(pth.dirname(pth_flows_normal))
    dump_data(flows_normal, file_out=pth_flows_normal)
    dump_data(flows_abnormal, file_out=pth_flows_abnormal)

    # 3) get the interval from normal flows, and use it to split flows
    durations = [np.max(pkt_times) - np.min(pkt_times) for fids, pkt_times, pkts in flows_normal]
    interval = np.quantile(durations, q=q_flow)
    lg(f'interval: {interval}, when q_flow={q_flow}')

    # 4) get subflows
    flows_normal = flows2subflows(flows_normal, interval, data_name=data_name, abnormal=False)
    flows_abnormal = flows2subflows(flows_abnormal, interval, data_name=data_name, abnormal=True)

    # dump all-subflows
    pth_flows_normal = pth.join(dir_in, data_name, f'all-subflows-normal-q_{q_flow}.dat')
    pth_flows_abnormal = pth.join(dir_in, data_name, f'all-subflows-abnormal-q_{q_flow}.dat')
    if not pth.exists(pth.dirname(pth_flows_normal)): os.makedirs(pth.dirname(pth_flows_normal))
    dump_data(flows_normal, file_out=pth_flows_normal)
    dump_data(flows_abnormal, file_out=pth_flows_abnormal)

    return flows_normal, flows_abnormal, interval


def choose_flows(flows, num=10000, random_state=42):
    num_flows = len(flows)
    num = num_flows if num > num_flows else num
    # idxs = np.random.choice(num_flows, size=num, replace=False)
    # abnormal_test_idx = np.in1d(range(abnormal_data.shape[0]), abnormal_test_idx)
    flows = shuffle(flows, random_state=random_state)

    return flows[:num]


def main(header, q_feat=0.9):
    """Get results on all data with the current parameters(header, ndm, gs, kjl)

    Parameters
    ----------
    header

    Returns
    -------
        0: succeed
        otherwise: failed.
    """
    lg(f'header-{header}-q_feat:{q_feat}')
    # dir_in = f'data_{ndm}'
    dir_in = f'data/data_IoT23/pcaps'
    dir_out = f'data/data_IoT23/csvs'  # csvs (normal and abnormal)
    datasets = [
        #
        # "CTU-IoT-Malware-Capture-1-1",
        # "CTU-IoT-Malware-Capture-3-1",
        # 'CTU-Honeypot-Capture-5-1'  # two normals
        # "CTU-IoT-Malware-Capture-7-1",
        # "CTU-IoT-Malware-Capture-8-1",
        # 'CTU-IoT-Malware-Capture-9-1', # costs too much time
        'CTU-IoT-Malware-Capture-20-1',
        # 'CTU-IoT-Malware-Capture-33-1', # costs too much time
        'CTU-IoT-Malware-Capture-48-1',
        'CTU-IoT-Malware-Capture-49-1',
    ]
    results = {}
    q_flow = 0.9  # It's used to split flows into subflows
    for data_name in datasets:
        prefix_pth = pth.join(dir_out, data_name)
        lg(prefix_pth)
        try:
            # 1. pcap to flows
            pth_flows_normal = pth.join(prefix_pth, 'flows_normal.dat')
            pth_flows_abnormal = pth.join(prefix_pth, 'flows_abnormal.dat')
            if not pth.exists(pth_flows_normal) or not pth.exists(pth_flows_abnormal):
                lg(f'one of .dat ({pth_flows_normal} or {pth_flows_abnormal}) does not exist, so we regenerate both.')

                data_type = 'mix' if 'CTU-IoT-Malware-Capture' in data_name else 'ind'
                flows_normal, flows_abnormal, interval = get_subflows(dir_in, data_name, data_type, q_flow)
                lg(f'len(flows_normal): {len(flows_normal)}, len(flows_abnormal): {len(flows_abnormal)}')

                # 2) only choose random 10,000 normal flows
                flows_normal = choose_flows(flows_normal, num=10000)
                # num = 600 if 'DS60_UChi_IoT' in data_name else 400  # ?
                flows_abnormal = choose_flows(flows_abnormal, num=600)
                lg(f'dump the data (subflows selected), and use them to get features data')
                lg(f'flows_normal: {len(flows_normal)}, flows_abnormal: {len(flows_abnormal)}')
                if not pth.exists(pth.dirname(pth_flows_normal)): os.makedirs(pth.dirname(pth_flows_normal))
                dict_normal = {'flows_normal': flows_normal, 'interval': interval, 'q_flow': q_flow}
                dump_data(dict_normal, file_out=pth_flows_normal)
                dict_abnormal = {'flows_abnormal': flows_abnormal, 'interval': interval, 'q_flow': q_flow}
                dump_data(dict_abnormal, file_out=pth_flows_abnormal)
                if pth.exists(pth_flows_normal) and pth.exists(pth_flows_abnormal):
                    lg(pth_flows_normal)
                    lg(pth_flows_abnormal)
                else:
                    msg = f'{pth_flows_normal},{pth_flows_abnormal} not exist'
                    raise FileExistsError(msg)
            else:
                lg(f'unpickle data')
                dict_normal = unpickle_data(pth_flows_normal)
                interval = dict_normal['interval']
                flows_normal = dict_normal['flows_normal']
                q_flow = dict_normal['q_flow']

                dict_abnormal = unpickle_data(pth_flows_abnormal)
                flows_abnormal = dict_abnormal['flows_abnormal']

                # modify dumped data if possible
                # flows_normal = unpickle_data(pth_flows_normal)
                # flows_abnormal = unpickle_data(pth_flows_abnormal)
                # interval= 69.55
                # dict_normal = {'flows_normal': flows_normal, 'interval': interval, 'q_flow': q_flow}
                # dump_data(dict_normal, out_file=pth_flows_normal)
                # dict_abnormal = {'flows_abnormal': flows_abnormal, 'interval': interval, 'q_flow': q_flow}
                # dump_data(dict_abnormal, out_file=pth_flows_abnormal)
                # break
        except Exception as e:
            msg = f'{get_subflows.__name__}, error: {e}'
            le(msg)
            continue

        result_each = {'q_flow': q_flow, 'interval': interval,
                       'q_feat': q_feat}  # store results obtained from the current dataset
        lg(f'data_name: {data_name}, result_each: {result_each}')
        for feat_set in FEAT_SETS:
            if 'fft' in feat_set:
                continue
            # 2. extract features from flows
            try:
                lg(f'feat_set: {feat_set}')
                if 'samp' in feat_set:
                    result = {}
                    fft_result = {}
                    for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                        # key_fft_pth = pth.join(dir_out, data_name, f"fft_{feat_set}", f'header:{str(header)}')
                        lg(f'feat_set: {feat_set}, q_samp: {q_samp}')
                        _result, _fft_result = get_each_intf(flows_normal, flows_abnormal, feat_set, q_feat, q_samp,
                                                             header,
                                                             prefix_pth)
                        result[q_samp] = _result
                        fft_result[q_samp] = _fft_result
                    result_each[feat_set] = result
                    result_each[f'fft_{feat_set}'] = fft_result
                else:
                    result, fft_result = get_each_intf(flows_normal, flows_abnormal, feat_set, q_feat, None, header,
                                                       prefix_pth)
                    result_each[feat_set] = result  # save feat_set
                    result_each[f'fft_{feat_set}'] = fft_result  # save fft_feat_set

            except Exception as e:
                msg = f'{get_each_intf.__name__}, {feat_set}, error: {e}'
                le(msg)
                result_each[feat_set] = {}

        # 3. store results
        # results generated by current parameters (header, ndm, gs, kjl) on all datasets will be stored at
        # the same file
        # -q_flow:{q_flow}-interval:{interval}
        file_out = pth.join(dir_out, data_name, f'all-features-header:{str(header)}')
        if not pth.exists(pth.dirname(file_out)): os.makedirs(pth.dirname(file_out))
        dump_data(result_each, file_out + '.dat')

        results[(dir_out, data_name)] = result_each  # results[data_name]=result_each

    file_out = pth.join(dir_out, f'datasets-header:{str(header)}.dat')
    if not pth.exists(pth.dirname(file_out)): os.makedirs(pth.dirname(file_out))
    dump_data(results, file_out)
    lg(f'file_out: {file_out}')

    return 0


def parse_cmd():
    """Parse commandline parameters

    Returns:
        args: parsed commandline parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--header", help="header", default=False, type=bool)
    parser.add_argument("-q", "--quant", help="quantile for fixing feature size", default=0.9, type=float)
    parser.add_argument("-t", "--time", help="start time of the application",
                        default=time.strftime(TIME_FORMAT, time.localtime()))
    args = parser.parse_args()
    lg(f"args: {args}")

    return args


if __name__ == '__main__':
    args = parse_cmd()
    header = args.header
    q_feat = args.quant
    main(header, q_feat)
