import pickle
from collections import OrderedDict

from scapy.layers.inet import IP

from itod.pparser.pcap import filter_ip, PcapReader
from itod.utils.tool import get_file_path


def dump_data(data, output_file='a.dat'):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
        print(data)
    return output_file


def load_data(output_file):
    with open(output_file, 'rb') as f:
        data = pickle.load(f)
        print(data)
    return data


def demo():
    #
    output_file = '/Users/kunyang/PycharmProjects/itod/examples/data/reprst/UCHI/IOT_2019/smtv_10.42.0.1/-subflow_interval=None_q_flow_duration=0.9'
    output_file = '/Users/kunyang/PycharmProjects/itod/examples/data/reprst/UCHI/IOT_2019/smtv_10.42.0.1/all-features-header:False.dat'
    # # data = ([1,2], [1,3], 'a')
    # # dump_data(data, out_file='a.dat')
    load_data(output_file)


def get_stat(pth_pcap, out_file='.dat'):
    print(f'pcap_file: {pth_pcap}')
    sessions = OrderedDict()  # key order of fid by time
    num_pkts = 0
    "filter pcap_file only contains the special srcIP "
    stat = {}

    # sessions= rdpcap(pcap_file).sessions()
    # res = PcapReader(pcap_file).read_all(count=-1)
    # from scapy import plist
    # sessions = plist.PacketList(res, name=os.path.basename(pcap_file)).sessions()
    for i, pkt in enumerate(PcapReader(pth_pcap)):  # iteratively get packet from the pcap
        if i % 10000 == 0:
            print(f'i_pkt: {i}')
        try:
            if IP in pkt:
                key = pkt['IP'].src
                if key not in stat.keys():
                    stat[key] = 1
                else:
                    stat[key] += 1

                key = pkt.dst
                if key not in stat.keys():
                    stat[key] = 1
                else:
                    stat[key] += 1
        except Exception as e:
            print('Error', e)

    for i, (k, v) in enumerate(stat.items()):
        if v > 1000:
            print(f'i={i}, {k}: {v}\n')

    print(f'len(sessions) {len(stat.keys())}')

    dump_data(stat, out_file)

    return stat


def main_MAWI():
    is_stat = False
    if is_stat:
        import os
        # file_name = 'samplepoint-F_202007011400-src_dst_185.8.54.240.pcap'
        file_name = 'samplepoint-F_202007011400.pcap'
        pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name='MAWI/WIDE_2020',
                                  file_name=file_name)
        stat_file = pcap_file + '-stat.dat'
        print(stat_file)
        if not os.path.exists(stat_file):
            stat = get_stat(pcap_file, out_file=stat_file)
        else:
            stat = load_data(stat_file)

        stat = dict(sorted(stat.items(), key=lambda kv: kv[1], reverse=True))
        for i, (k, v) in enumerate(stat.items()):
            if v > 1000:
                print(f'i={i}, {k}: {v}\n')
        # print(stat)

    ips = ['23.222.78.164', '203.78.4.32', '203.78.8.151', '23.223.19.175', '114.234.20.197', '114.234.12.139']
    # ['202.119.210.242', '23.99.220.247', '203.78.23.227', '51.95.212.72', '163.98.16.76',
    # '92.206.43.252', '202.66.205.237', '202.75.33.206', '202.75.33.114', '167.50.204.117']
    direction = 'src_dst'
    file_name = f'samplepoint-F_202007011400.pcap'
    for v in ips:
        if direction == 'src_dst':
            #     # filter pcap: tshark -r samplepoint-F_202007011400.pcap -w samplepoint-F_202007011400-src_dst_202.119.210.242.pcap ip.addr=='202.119.210.242'
            pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name='MAWI/WIDE_2020',
                                      file_name=file_name)
        else:
            pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name='MAWI/WIDE_2020',
                                      file_name=file_name + f'-src_dst_{v}.pcap')
        out_file = get_file_path(ipt_dir='original_data/reprst',
                                 dataset_name='MAWI/WIDE_2020',
                                 file_name=file_name + f'-{direction}_{v}.pcap')
        print(pcap_file, out_file)
        filter_ip(pcap_file, out_file, ips=[v], direction=direction, verbose=20)


def main_UNB():
    is_stat = False
    dataset_name = 'UNB/CICIDS_2017/pcaps'
    if is_stat:
        import os
        file_name = 'Monday/Monday-WorkingHours.pcap'
        pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name=dataset_name,
                                  file_name=file_name)
        stat_file = pcap_file + '-stat.dat'
        print(stat_file)
        if not os.path.exists(stat_file):
            stat = get_stat(pcap_file, out_file=stat_file)
        else:
            stat = load_data(stat_file)

        stat = dict(sorted(stat.items(), key=lambda kv: kv[1], reverse=True))
        for i, (k, v) in enumerate(stat.items()):
            if v > 10000:
                print(f'i={i}, {k}: {v}\n')
        # print(stat)

    ips = ['23.15.4.24', '192.168.10.12', '192.168.10.9', '192.168.10.16', '192.168.10.51']
    # ['202.119.210.242', '23.99.220.247', '203.78.23.227', '51.95.212.72', '163.98.16.76',
    # '92.206.43.252', '202.66.205.237', '202.75.33.206', '202.75.33.114', '167.50.204.117']
    direction = 'src_dst'
    file_name = 'Monday/Monday-WorkingHours.pcap'
    for v in ips:
        if direction == 'src_dst':
            #     # filter pcap: tshark -r samplepoint-F_202007011400.pcap -w samplepoint-F_202007011400-src_dst_202.119.210.242.pcap ip.addr=='202.119.210.242'
            pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name=dataset_name,
                                      file_name=file_name)
        else:
            pcap_file = get_file_path(ipt_dir='original_data/reprst', dataset_name=dataset_name,
                                      file_name=file_name + f'-src_dst_{v}.pcap')
        out_file = get_file_path(ipt_dir='original_data/reprst',
                                 dataset_name=dataset_name,
                                 file_name=file_name + f'-{direction}_{v}.pcap')
        print(pcap_file, out_file)
        filter_ip(pcap_file, out_file, ips=[v], direction=direction, verbose=20)


if __name__ == '__main__':
    main_UNB()
