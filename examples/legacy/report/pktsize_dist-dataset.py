"""
     for both MAWI, SFrig and UNB(PC1) ?
- histgram for packet sizes
-

"""
import os, sys
import pickle
from collections import Counter, OrderedDict
import pprint

import sklearn
from scapy.layers.inet import TCP, UDP
from scapy.utils import PcapReader
from sklearn.model_selection import train_test_split

from itod.pparser.pcap import session_extractor

lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set()

pprint.sorted = lambda x, key=None: x

# from itod.ndm.train_test import extract_data, preprocess_data
from itod.utils.utils import dump_data

# 0.4 get root logger
# from itod import log

# lg = log.get_logger(name=None, level='DEBUG', out_dir='./log/data_kjl', app_name='correlation')

datasets = [
    # Naming: (department/dataname_year/device)
    'UNB/CICIDS_2017/pc_192.168.10.5',
    # 'UNB/CICIDS_2017/pc_192.168.10.8',
    # # 'UNB/CICIDS_2017/pc_192.168.10.9',
    'UNB/CICIDS_2017/pc_192.168.10.14',
    # # 'UNB/CICIDS_2017/pc_192.168.10.15',
    # # #
    # 'CTU/IOT_2017/pc_10.0.2.15',
    # #
    'MAWI/WIDE_2019/pc_202.171.168.50',
    # # 'MAWI/WIDE_2020/pc_203.78.7.165',
    # # # # # #
    'UCHI/IOT_2019/smtv_10.42.0.1',
    # #
    # # 'UCHI/IOT_2019/ghome_192.168.143.20',
    # # 'UCHI/IOT_2019/scam_192.168.143.42',
    'UCHI/IOT_2019/sfrig_192.168.143.43',
    'UCHI/IOT_2019/bstch_192.168.143.48'

]  # 'DEMO_IDS/DS-srcIP_192.168.10.5'


def _get_each_correlation(x, y):
    rho = np.corrcoef(x, y)[0, 1]
    rho = 0 if np.isnan(rho) else rho
    return rho


def get_data(in_file='.dat', feat_set='iat_size'):
    with open(in_file, 'rb') as f:
        inst = pickle.load(f)
    X_train, y_train, X_test, y_test = inst.dataset_inst.dataset_dict[f'{feat_set}_dict']['data']

    size = 5000
    if len(y_train) > size:
        X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

    # sd_inst = Dataset()
    # X_train, X_test = sd_inst.normalise_data(X_train, X_test,
    #                                          norm_method=self.params['norm_method'])
    # select a part of val_data
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                    stratify=y_test,
                                                    random_state=42)
    print(f'X_train: {X_train.shape}, y_train: {Counter(y_train)}')
    print(f'X_val: {X_val.shape}, y_val: {Counter(y_val)}')
    print(f'X_test: {X_test.shape}, y_test: {Counter(y_test)}')

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_size(X_test, y_test):
    """normal and abnormal have the same number of flows, but different number of packets"""
    normal_size = []
    abnormal_size = []
    for x, y in zip(X_test, y_test):
        x = [v for v in x if v > 0]  # v ==0, which is the append 0 for the packet size.
        if y == 0:
            normal_size.extend(x)
        else:
            abnormal_size.extend(x)

    qs = [0.0, 0.3, 0.5, 0.9, 1]
    print(f'normal:{np.quantile(normal_size, q=qs)}')
    print(f'abnormal:{np.quantile(abnormal_size, q=qs)}')

    return normal_size, abnormal_size


def get_packet_size_in_flows(X_test, y_test, MTU=1500):
    """normal and abnormal have the same number of flows, but different number of packets"""

    n_normal_flows = 0
    n_abnormal_flows = 0

    n_normal_pkts = 0
    n_abnormal_pkts = 0

    n_normal_big_pkts = 0
    n_abnormal_big_pkts = 0

    n_normal_big_flows = 0
    n_abnormal_big_flows = 0

    for x, y in zip(X_test, y_test):
        x = [v for v in x if v > 0]
        if y == 0:
            n_normal_flows += 1
            n_normal_pkts += len(x)
        else:
            n_abnormal_flows += 1
            n_abnormal_pkts += len(x)

    for x, y in zip(X_test, y_test):
        x = [v for v in x if v > MTU]  # v ==0, which is the append 0 for the packet size.
        if x != [] or len(x) != 0:  # flow has pkt_size > MTU
            if y == 0:
                n_normal_big_flows += 1
                n_normal_big_pkts += len(x)
            else:
                n_abnormal_big_flows += 1
                n_abnormal_big_pkts += len(x)

    res = {'normal': {'pkts': n_normal_pkts,
                      'flows': n_normal_flows,
                      'big_pkts': n_normal_big_pkts,
                      'big_flows': n_normal_big_flows,
                      'ratio_pkts': f"{n_normal_big_pkts / n_normal_pkts:.4f}" if n_normal_pkts != 0 else 0,
                      'ratio_flows': f"{n_normal_big_flows / n_normal_flows:.4f}" if n_normal_flows != 0 else 0,
                      },
           'abnormal': {'pkts': n_abnormal_pkts,
                        'flows': n_abnormal_flows,
                        'big_pkts': n_abnormal_big_pkts,
                        'big_flows': n_abnormal_big_flows,
                        'ratio_pkts': f"{n_abnormal_big_pkts / n_abnormal_pkts:.4f}" if n_abnormal_pkts != 0 else 0,
                        'ratio_flows': f"{n_abnormal_big_flows / n_abnormal_flows:.4f}" if n_abnormal_flows != 0 else 0,
                        },
           }
    return res


def get_pktsize(in_dir='', out_dir='out', feat_type='size', header=False):
    # datasets = ['DS10_UNB_IDS/DS11-srcIP_192.168.10.5', # UNB(PC1)
    #             'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50', # MAWI
    #             'DS60_UChi_IoT/DS63-srcIP_192.168.143.43'   # SFrig
    #             ]
    results = {}
    MTU = 1500 + 14  # (mac length)
    for i, dataset in enumerate(datasets):
        key_pth = os.path.join(in_dir, dataset, feat_type, f"header:{header}")
        print(f'i: {i}, key_path: {key_pth}')
        pth_pcap = os.path.join(in_dir, dataset, f'AGMT-WorkingHours/pc_192.168.10.5_AGMT.pcap')
        pcap2flows(pth_pcap)
        # in_file = os.path.join(in_dir, dataset, f'all-features-header:{header}.dat')
        # # in_file = os.path.join(in_dir, dataset, f'-subflow_interval=None_q_flow_duration=0.9')
        # X_train, y_train, X_val, y_val, X_test, y_test = get_data(in_file, feat_type)
        # # 2 get flow_size:
        # res = {}
        # res['train'] = get_packet_size_in_flows(X_train, y_train, MTU=MTU)
        # # res['val'] = get_packet_size_in_flows(X_val, y_val, MTU=MTU)
        # res['test'] = get_packet_size_in_flows(X_test, y_test, MTU=MTU)
        # # normal, abnormal = get_size(X_test, y_test)
        # # pkt_sizes = {'normal': normal, 'abnormal': abnormal}
        # # results[(key_pth, dataset, feat_set, X_test.shape)] = pkt_sizes
        # # print(res)
        # pprint.pprint(res)
        # # pprint(res, compact=False, sort_dicts=True)       # only works in python3.8+
        # print('\n')
        # results[dataset] = res

    # # save all results
    # out_file = os.path.splitext(out_file)[0] + '_all.dat'
    # print(f'out_file: {out_file}')
    # dump_data(results, out_file)

    return results


def get_pktsize_kjl(in_dir='', out_dir='out', feat_type='size', header=False):
    # datasets = ['DS10_UNB_IDS/DS11-srcIP_192.168.10.5', # UNB(PC1)
    #             'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50', # MAWI
    #             'DS60_UChi_IoT/DS63-srcIP_192.168.143.43'   # SFrig
    #             ]
    results = {}
    MTU = 1500
    for i, dataset in enumerate(datasets):
        key_pth = os.path.join(in_dir, dataset, feat_type, f"header:{header}")
        print(f'i: {i}, key_path: {key_pth}')
        in_file = os.path.join(in_dir, dataset, f'all-features-header:{header}.dat')
        # in_file = os.path.join(in_dir, dataset, f'-subflow_interval=None_q_flow_duration=0.9')
        X_train, y_train, X_val, y_val, X_test, y_test = get_data(in_file, feat_type)
        # 2 get flow_size:
        res = {}
        res['train'] = get_packet_size_in_flows(X_train, y_train, MTU=MTU)
        # res['val'] = get_packet_size_in_flows(X_val, y_val, MTU=MTU)
        res['test'] = get_packet_size_in_flows(X_test, y_test, MTU=MTU)
        # normal, abnormal = get_size(X_test, y_test)
        # pkt_sizes = {'normal': normal, 'abnormal': abnormal}
        # results[(key_pth, dataset, feat_set, X_test.shape)] = pkt_sizes
        # print(res)
        pprint.pprint(res)
        # pprint(res, compact=False, sort_dicts=True)       # only works in python3.8+
        print('\n')
        results[dataset] = res

    # # save all results
    # out_file = os.path.splitext(out_file)[0] + '_all.dat'
    # print(f'out_file: {out_file}')
    # dump_data(results, out_file)

    return results


def plot_pktsize_multi(results, out_dir, title=None, show=True):
    # # only show the top 4 figures
    datasets = [
        ('UNB/CICIDS_2017/pc_192.168.10.5', 'UNB(PC1)'),  # data_name is unique
        ('UNB/CICIDS_2017/pc_192.168.10.14', 'UNB(PC4)'),
        # ('CTU/IOT_2017/pc_10.0.2.15', 'CTU'),
        ('MAWI/WIDE_2019/pc_202.171.168.50', 'MAWI'),
        ('UCHI/IOT_2019/smtv_10.42.0.1', 'TV&RT'),
        ('UCHI/IOT_2019/sfrig_192.168.143.43', 'SFrig'),
        ('UCHI/IOT_2019/bstch_192.168.143.48', 'BSTch'),
    ]
    new_results = {}
    for i, (dataset, name) in enumerate(datasets):
        for j, (key, result) in enumerate(results.items()):
            _key_path, _dataset, _feat_set, X_test_shape = key
            if dataset in key:
                new_results[(_key_path, _dataset, name, _feat_set, X_test_shape)] = result
    t = 0
    cols = 2
    fontsize = 20
    ## http://jose-coto.com/styling-with-seaborn
    # colors = ["m", "#4374B3"]
    # palette = sns.color_palette('RdPu', 1)  # a list
    # palette = sns.color_palette('YlOrRd', 7)[:2]  # YlOrRd
    fig, axes = plt.subplots(3, cols, figsize=(18, 8))  # (width, height)
    # print(new_results)
    for i, (key, res) in enumerate(new_results.items()):
        # print(f"i: {i}, {key}, res: {res}")  # hue = feat_set
        key_path, dataset, short_name, feat_set, X_test_shape = key
        # # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
        # # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
        # HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']

        # data = sorted(range(len(corrs)), key=lambda i: abs(corrs[i]), reverse=True)[:6]  # top 6 values
        # # data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in sorted(data, reverse=False)]
        # data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in data]
        # print(f"i: {i}, {key}, corrs: {data}")

        # new_yerrs = [1 / (np.sqrt(X_test_shape[0]))] * 6  # for err_bar
        # print(f'i: {i}, {new_yerrs}')
        if short_name == 'UNB(PC1)':
            max_size = 300
        elif short_name == 'UNB(PC4)':
            max_size = 300
        elif short_name == 'MAWI':
            max_size = 80
        elif short_name == 'TV&RT':
            max_size = 300
        elif short_name == 'SFrig':
            max_size = 300
        elif short_name == 'BSTch':
            max_size = 500
        else:
            max_size = 1000
        x1 = res['normal']
        x1 = [v for v in x1 if v < max_size]
        y1 = len(x1) * [0]
        x2 = res['abnormal']
        x2 = [v for v in x2 if v < max_size]
        y2 = len(x2) * [1]
        x = x1 + x2
        y = y1 + y2
        print(f'normal: {len(x1)}, y1: {len(y1)}, abnormal: {len(x2)}, y2: {len(y2)}')
        assert len(x) == len(y)
        qs = [0.0, 0.3, 0.5, 0.9, 1]
        print(f'normal:{np.quantile(x1, q=qs)}')
        print(f'abnormal:{np.quantile(x2, q=qs)}')
        res = np.asarray(list(zip(x, y)))
        print(f'len(zip(x,y)): {len(res)}')
        df = pd.DataFrame(res, columns=[f'x', 'y'])
        if i % cols == 0 and i > 0:
            t += 1
        # g = sns.histplot(x=f"Xi_y", y="corr_rho", ax=axes[t, i % cols], hue='feat_set', data=df,
        #                 palette=palette)  # palette=palette,

        g = sns.histplot(data=df, x="x", hue="y", ax=axes[t, i % cols], bins=50, stat='density')  # 'density'

        g.set(xlabel=None)
        # g.set(ylim=(0, 0.2))
        if i % cols == 0:  # i % cols == 0
            # g.set_ylabel(r'$\rho$', fontsize=fontsize + 4)
            g.set_ylabel(r'Density', fontsize=fontsize + 4)
            # print(g.get_yticks())
            # g.set_yticks([-1, -0.5, 0, 0.5, 1])
            # g.set_yticklabels(g.get_yticks(), fontsize=fontsize + 2)  # set the number of each value in y axis
            g.set_yticklabels([float(f'{v:.2}') for v in g.get_yticks()],
                              fontsize=fontsize + 2)
            # print(g.get_yticks())
        else:
            g.set(ylabel=None)
            g.set_yticklabels(['' for v_tmp in g.get_yticks()])
            # g.set_yticklabels(np.asarray(g.get_yticks(), dtype=int),
            #                   fontsize=fontsize + 2)  # set the number of each value in y axis
            g.set_yticklabels([float(f'{v:.2}') for v in g.get_yticks()],
                              fontsize=fontsize + 2)  # set the number of each value in y axis
            g.set_ylabel('')

        # g.set_title(dataset_name)
        g.get_legend().set_visible(False)
        g.set_xticklabels(np.asarray(g.get_xticks(), dtype=int), fontsize=fontsize + 0, rotation=0, ha="center")

        # ys = []
        # xs = []
        # width = 0
        # for i_p, p in enumerate(g.patches):
        #     height = p.get_height()
        #     width = p.get_width()
        #     ys.append(height)
        #     xs.append(p.get_x())
        #     if i_p == 0:
        #         pre = p.get_x() + p.get_width()
        #     if i_p > 0:
        #         cur = p.get_x()
        #         g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
        #         pre = cur + p.get_width()
        #     ## https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
        #     p.set_width(width / 3)  # set the bar width
        #     # we recenter the bar
        #     p.set_x(p.get_x() + width / 3)
        g.set_title(short_name, fontsize=fontsize + 8)

        # # add error bars
        # g.errorbar(x=xs + width / 2, y=ys,
        #            yerr=new_yerrs, fmt='none', c='b', capsize=3)

    # # get the legend and modify it
    # handles, labels = g.get_legend_handles_labels()
    fig.legend(['novelty', 'normal'], title='Packet size', title_fontsize=fontsize, loc='lower center', ncol=2,
               prop={'size': fontsize - 2})  # loc='lower right',  loc = (0.74, 0.13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # title = dataset
    # # fig.suptitle("\n".join(["a big long suptitle that runs into the title"]*2), y=0.98)
    # fig.suptitle(title)
    out_file = os.path.join(out_dir, feat_set, "header:True", 'packet_size-bar.pdf')
    if not os.path.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
    print(out_file)
    plt.savefig(out_file)  # should use before plt.show()
    if show: plt.show()
    plt.close(fig)
    plt.close("all")
    # sns.reset_orig()
    # sns.reset_defaults()


#
#
# def plot_pktsize(results, out_dir, title=None, show=True):
#     for i, (key, results) in enumerate(results.items()):
#         print(f"i: {i}, {key}")  # hue = feat_set
#         key_path, dataset, feat_set, X_test_shape = key
#         # # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
#         # # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
#         # HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']
#         # data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in range(9)]
#
#         cols = 2
#         fig, axes = plt.subplots(1, cols, figsize=(18, 8))  # (width, height)
#         axes = axes.reshape(-1, cols)
#
#         num_bins = 30
#         normal, abnormal = results['normal'], results['abnormal']
#         # x_normal = [sum(v) for v in normal]
#         # x_abnormal = [sum(v) for v in abnormal]
#         x_normal = normal
#         x_abnormal = abnormal
#         n, bins, patches = axes[0,0].hist(x_normal, num_bins, density=False, facecolor='g', alpha=0.75)
#         n, bins, patches = axes[0, 1].hist(x_abnormal, num_bins, density=False, facecolor='g', alpha=0.75)
#         flg.xlabel('Size')
#         # plt.ylabel('Count')
#         # plt.title('')
#         # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#         # plt.xlim(40, 160)
#         # plt.ylim(0, 0.03)
#         # plt.grid(True)
#         # plt.show()
#
#
#         # fig = plt.figure(figsize=(10, 5))  # (width, height)
#         #
#         # df = pd.DataFrame(data, columns=[f'Xi_y', 'feat_set', 'corr_rho'])
#         # g = sns.barplot(x=f"Xi_y", y="corr_rho", hue='feat_set', data=df)  # palette=palette,
#         # g.set(xlabel=None)
#         # g.set(ylabel='Rho')
#         # g.set_ylim(-1, 1)
#         # # g.set_title(dataset_name)
#         # g.get_legend().set_visible(True)  # False
#         # g.set_xticklabels(g.get_xticklabels(), fontsize=12, rotation=30, ha="center")
#         #
#         # ys = []
#         # xs = []
#         # width = 0
#         # for i_p, p in enumerate(g.patches):
#         #     height = p.get_height()
#         #     ys.append(height)
#         #     xs.append(p.get_x())
#         #     if i_p == 0:
#         #         pre = p.get_x() + p.get_width()
#         #     if i_p > 0:
#         #         cur = p.get_x()
#         #         g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
#         #         pre = cur + p.get_width()
#         # plt.legend(loc='upper right')
#         # # # get the legend and modify it
#         # # handles, labels = g.get_legend_handles_labels()
#         # # fig.legend(handles, labels, title=None, loc='upper right', ncol=1,
#         # #  prop={'size': 8})  # loc='lower right',  loc = (0.74, 0.13)
#
#         plt.tight_layout()
#
#         title = dataset
#         # g.set_title(title + ' (header:True)', fontsize=13)
#         out_file = os.path.join(out_dir, dataset, feat_set, "header:False", '-size-bar.png')
#         print(out_file)
#         if not os.path.exists(os.path.dirname(out_file)):
#             os.makedirs(os.path.dirname(out_file))
#         plt.savefig(out_file)  # should use before plt.show()
#         if show: plt.show()
#         plt.close(fig)
#         plt.close("all")
#
#

def pcap2flows(pth_pcap, num_pkt_thresh=2, verbose=True):
    '''Reads pcap and divides packets into 5-tuple flows (arrival times and sizes)

           Arguments:
             pcap_file (string) = path to pcap file
             num_pkt_thresh (int) = discards flows with fewer packets than max(2, thresh)

           Returns:
             flows (list) = [(fid, arrival times list, packet sizes list)]
        '''

    print(f'pcap_file: {pth_pcap}')
    sessions = OrderedDict()  # key order of fid by time
    num_pkts = 0
    pkt_sizes = []
    "filter pcap_file only contains the special srcIP "
    i = 0
    FRAME_LEN = 1500 + 14
    cnt = 0
    try:
        # sessions= rdpcap(pcap_file).sessions()
        # res = PcapReader(pcap_file).read_all(count=-1)
        # from scapy import plist
        # sessions = plist.PacketList(res, name=os.path.basename(pcap_file)).sessions()
        for i, pkt in enumerate(PcapReader(pth_pcap)):  # iteratively get packet from the pcap
            if i % 10000 == 0:
                print(f'i_pkt: {i}')
            # sess_key = session_extractor(pkt)  # this function treats bidirection as two sessions.
            # if ('TCP' in sess_key) or ('UDP' in sess_key) or (6 in sess_key) or (17 in sess_key):
            if (TCP in pkt) or (UDP in pkt):
                if len(pkt) > FRAME_LEN:
                    cnt += 1
                    print(
                        f'i: {i}, pkt.wirelen: {pkt.wirelen}, len(pkt): {len(pkt)}, pkt.payload.len: {pkt.payload.len}')
                pkt_sizes.append(len(pkt))

    except Exception as e:
        print('Error', e)
    print(f'tot_pkts: {i + 1}, {cnt} packets have wirelen > FRAME_LEN({FRAME_LEN})')
    print(f'len(sessions) {len(sessions.keys())}')

    return pkt_sizes


def main1():
    pkt_sizes = pcap2flows(
        pth_pcap='data/reprst_srcip/UCHI/IOT_2019/sfrig_192.168.143.43/fridge_cam_sound_ghome_2daysactiv-sfrig_normal.pcap-192.168.143.43.pcap')
    qs = [0.0, 0.3, 0.5, 0.9, 0.98, 1]
    print(f'pkt_sizes:{np.quantile(pkt_sizes, q=qs)}')


# main1()


def seperate_normal_abnormal(X, y, random_state=42):
    X = np.asarray(X, dtype=float)
    y = [0 if v.startswith('normal') else 1 for i, v in enumerate(list(y))]
    y = np.asarray(y)

    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    # normal_idx = [i for i, v in enumerate(y) if v.startswith('normal')]
    # abnormal_idx =  [i for i, v  in enumerate(y) if v.startswith('abnormal')]

    X_normal, y_normal = X[normal_idx], y[normal_idx]
    X_abnormal, y_abnormal = X[abnormal_idx], y[abnormal_idx]

    print(f"X.shape: {X.shape}, y: {Counter(y)}")

    return X_normal, y_normal, X_abnormal, y_abnormal


def split_left_test(X_normal, y_normal, X_abnormal, y_abnormal, test_size=600, random_state=42):
    """Split train and test set

    Parameters
    ----------
    X
    y
    show

    Returns
    -------

    """
    # n_val_normal = 300 if len(y_abnormal) > 300 else int(len(y_abnormal) * 0.8)

    X_normal, y_normal = sklearn.utils.shuffle(X_normal, y_normal, random_state=random_state)
    X_abnormal, y_abnormal = sklearn.utils.shuffle(X_abnormal, y_abnormal, random_state=random_state)

    n_abnormal = int(test_size // 2)
    # n_abnormal = 300 if len(y_abnormal) > n_abnormal * 2  else 300

    # X_normal, y_normal = sklearn.utils.resample(X, y, n_samples=n_abnormal, random_state=random_state, replace=False)
    # X_normal, y_normal = sklearn.utils.resample(X, y, n_samples=n_abnormal, random_state=random_state,
    #                                             replace=False)
    ######## get test data
    X_test_normal = X_normal[:n_abnormal, :]
    y_test_normal = y_normal[:n_abnormal].reshape(-1, 1)

    X_test_abnormal = X_abnormal[:n_abnormal, :]
    y_test_abnormal = y_abnormal[:n_abnormal].reshape(-1, 1)
    X_test = np.vstack((X_test_normal, X_test_abnormal))
    # normal and abnormal have the same size in the test set
    y_test = np.vstack((y_test_normal, y_test_abnormal)).flatten()

    ########## left data
    X_normal = X_normal[n_abnormal:, :]
    y_normal = y_normal[n_abnormal:]
    X_abnormal = X_abnormal[n_abnormal:, :]
    y_abnormal = y_abnormal[n_abnormal:]
    y_left = np.vstack((y_normal.reshape(-1, 1), y_abnormal.reshape(-1, 1))).flatten()

    print(f"left: {Counter(y_left)}, test: {Counter(y_test)}")

    return X_normal, y_normal, X_abnormal, y_abnormal, X_test, y_test


def split_train_val(X_normal, y_normal, X_abnormal, y_abnormal, train_size=5000, val_size=100, random_state=42):
    """Split train and test set

    Parameters
    ----------
    X
    y
    show

    Returns
    -------

    """
    X_normal, y_normal = sklearn.utils.shuffle(X_normal, y_normal, random_state=random_state)
    X_abnormal, y_abnormal = sklearn.utils.shuffle(X_abnormal, y_abnormal, random_state=random_state)

    n_train_normal = train_size
    n_val_normal = int(val_size // 2)
    ######## get train data
    X_train_normal = X_normal[:n_train_normal, :]
    y_train_normal = y_normal[:n_train_normal].reshape(-1, 1)
    X_train = X_train_normal
    y_train = y_train_normal.flatten()

    n_val_normal = n_val_normal if len(y_abnormal) > n_val_normal else int(len(y_abnormal) * 0.8)

    X_val_normal = X_normal[n_train_normal: n_train_normal + n_val_normal, :]
    y_val_normal = y_normal[n_train_normal:n_train_normal + n_val_normal].reshape(-1, 1)
    X_val_abnormal = X_abnormal[:n_val_normal, :]
    y_val_abnormal = y_abnormal[:n_val_normal].reshape(-1, 1)
    X_val = np.vstack((X_val_normal, X_val_abnormal))
    # normal and abnormal have the same size in the test set
    y_val = np.vstack((y_val_normal, y_val_abnormal)).flatten()

    print(f"train: {Counter(y_train)}, val: {Counter(y_val)}")

    return X_train, y_train, X_val, y_val


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data


def get_pktsize_kjl(in_dir='', out_dir='out', feat_type='size', header=False):
    datasets = ['UNB24', 'CTU1', 'MAWI1_2020', "ISTS1", "MACCDC1", "SFRIG1_2020", "AECHO1_2020"]
    results = {}
    MTU = 1500 + 14  # (mac length)

    def get_data(in_file, feat_type, random_state=42):
        X, y = load_data(in_file)
        X_normal, y_normal, X_abnormal, y_abnormal = seperate_normal_abnormal(X, y, random_state=random_state)
        # get the unique test set
        X_normal, y_normal, X_abnormal, y_abnormal, X_test, y_test = split_left_test(X_normal, y_normal, X_abnormal,
                                                                                     y_abnormal, test_size=600,
                                                                                     random_state=random_state)

        X_train, y_train, X_val, y_val = split_train_val(X_normal, y_normal, X_abnormal, y_abnormal,
                                                         train_size=5000, val_size=int(len(y_test) * 0.25),
                                                         random_state=(i + 1) * 100)

        return X_train, y_train, X_val, y_val, X_test, y_test

    for i, dataset in enumerate(datasets):
        key_pth = os.path.join(in_dir, dataset)
        print(f'i: {i}, key_path: {key_pth}')
        in_file = os.path.join(in_dir, dataset, f'Xy-normal-abnormal.dat')
        # in_file = os.path.join(in_dir, dataset, f'-subflow_interval=None_q_flow_duration=0.9')
        X_train, y_train, X_val, y_val, X_test, y_test = get_data(in_file, feat_type)
        # 2 get flow_size:
        res = {}
        res['train'] = get_packet_size_in_flows(X_train, y_train, MTU=MTU)
        # res['val'] = get_packet_size_in_flows(X_val, y_val, MTU=MTU)
        res['test'] = get_packet_size_in_flows(X_test, y_test, MTU=MTU)
        # normal, abnormal = get_size(X_test, y_test)
        # pkt_sizes = {'normal': normal, 'abnormal': abnormal}
        # results[(key_pth, dataset, feat_set, X_test.shape)] = pkt_sizes
        # print(res)
        pprint.pprint(res)
        # pprint(res, compact=False, sort_dicts=True)       # only works in python3.8+
        print('\n')
        results[dataset] = res

    # # save all results
    # out_file = os.path.splitext(out_file)[0] + '_all.dat'
    # print(f'out_file: {out_file}')
    # dump_data(results, out_file)

    return results


def main():
    # Represt
    direction = 'src'
    if direction == 'src':
        in_dir = 'data/reprst_srcip'
        out_dir = 'out/reprst_srcip/pktsize'
    elif direction == 'src_dst':
        in_dir = 'data/reprst'
        out_dir = 'out/reprst/pktsize'
    results = get_pktsize(in_dir, out_dir, feat_type='size')

    # # KJL
    # direction = 'src_dst'
    # if direction == 'src_dst':
    #     in_dir = f'../../kjl/examples/speedup/data/src_dst/iat_size-header_False'
    #     out_dir = 'out/reprst_srcip/pktsize'
    # # elif direction == 'src_dst':
    # #     in_dir = 'data/reprst'
    # #     out_dir = 'out/reprst/pktsize'
    # results = get_pktsize_kjl(in_dir, out_dir, feat_type='iat_size')

    # plot_pktsize_multi(results, out_dir, show=True)


if __name__ == '__main__':
    main()
