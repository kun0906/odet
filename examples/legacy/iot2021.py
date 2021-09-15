""" Analyze IOT datasets (data-clean.zip: 20GB, 20210714) collected on 2021.
    PYTHONPATH=./:./itod python3.7 examples/reprst/data/iot2021.py
"""
import collections
import os
import subprocess

import numpy as np

from odet.pparser.parser import PCAP

RANDOM_STATE = 42


class IOT2021(PCAP):

    def get_flows(self, in_file='xxx.pcap'):
        # flows: [(fid, arrival times list, packet sizes list)]
        self.flows, self.num_pkts = pcap2flows(in_file, num_pkt_thresh=2)

    def keep_ip(self, pcap_file, kept_ips=[], output_file=''):

        if output_file == '':
            output_file = os.path.splitext(pcap_file)[0] + 'kept_ips.pcap'  # Split a path in root and extension.
        # only keep srcIPs' traffic
        # srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
        # filter by mac srcIP address
        srcIP_str = " or ".join([f'eth.src=={srcIP}' for srcIP in kept_ips])
        cmd = f"tshark -r {pcap_file} -w {output_file} {srcIP_str}"

        print(f'{cmd}')
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        return output_file


def get_pcaps(in_dir, file_type='normal'):
    files = collections.defaultdict(list)
    for activity_name in sorted(os.listdir(in_dir)):
        if activity_name.startswith('.'): continue
        activity_dir = os.path.join(in_dir, activity_name)
        for partcipant_id in sorted(os.listdir(activity_dir)):
            if partcipant_id.startswith('.'): continue
            partcipant_dir = os.path.join(activity_dir, partcipant_id)
            for f in sorted(os.listdir(partcipant_dir)):
                if f.startswith('.'): continue
                if f.endswith('pcap'):
                    f = os.path.join(partcipant_dir, f)
                    files[activity_name].append(f)
                    # files.append(f)
                else:
                    pass
    return files


def get_mac_ip(flows):
    ips = []
    macs = []
    # [(fid, arrival times list, packet sizes list)]
    for i, (fid, pkt_times, pkts) in enumerate(flows):
        macs.append(pkts[0].src)
        ips.append(fid[0])
    print(set(ips))
    return macs, ips


def get_durations(flows):
    durations = []
    # [(fid, arrival times list, packet sizes list)]
    for i, (fid, pkt_times, pkts) in enumerate(flows):
        start = min(pkt_times)
        end = max(pkt_times)
        durations.append(end - start)

    return durations


# IP has changed (dynamic IPs) in different collection process, so please use mac to filter packets.
ip2device = {'192.168.143.152': 'refrigerator', }
device2ip = {'refrigerator': '192.168.143.43', 'nestcam': '192.168.143.104', 'alexa': '192.168.143.74'}
device2mac = {'refrigerator': '70:2c:1f:39:25:6e', 'nestcam': '18:b4:30:8a:9f:b2', 'alexa': '4c:ef:c0:0b:91:b3'}



def main(device='refrigerator'):
    in_dir = f'../Datasets/UCHI/IOT_2021/data-clean/{device}'
    out_dir = f'examples/datasets/IOT2021/data-clean/{device}'
    device_meta_file = os.path.join(out_dir, f'{device}.dat')
    device_meta = {}
    if not os.path.exists(device_meta_file):
        device_files = get_pcaps(in_dir, file_type='normal')
        for i, (activity_name, files) in enumerate(device_files.items()):
            activity_flows = []
            for j, f in enumerate(files):
                print(j, f)
                # create the PCAP object
                pp = IOT2021()

                # filter unnecesarry IP addresses
                filtered_f = os.path.join(out_dir,
                                          os.path.splitext(os.path.relpath(f, start=in_dir))[0] + '-filtered.pcap')
                check_path(filtered_f)
                # pp.keep_ip(f, kept_ips=[device2ip[device]], output_file=filtered_f)
                pp.keep_ip(f, kept_ips=[device2mac[device]], output_file=filtered_f)

                # parse pcap and get the flows (only forward flows (sent from src IP))
                pp.get_flows(filtered_f)

                # concatenated the flows to the total flows
                device_files[activity_name][j] = (filtered_f, pp.flows)
                activity_flows += pp.flows
                # break

            # activity_flows = sum(len(flows_) for flows_ in ])
            print(f'activity_flows: {len(activity_flows)}')
            device_meta[activity_name] = (activity_flows, device_files[activity_name])
        check_path(device_meta_file)
        print(device_meta_file)
        dump_data(device_meta, output_file=device_meta_file)
    else:
        device_meta = load_data(device_meta_file)

    ips = set()
    macs = set()
    for i, (activity_name, vs_) in enumerate(device_meta.items()):
        activity_flows, file_flows = vs_
        print(i, activity_name, len(activity_flows))
        macs_, ips_ = get_mac_ip(activity_flows)
        # print strange IP and pcap_file
        for v_, (f_, _) in zip(ips_, file_flows):
            if v_ == '0.0.0.0':
                print(activity_name, v_, f_)
        macs.update(macs_)
        ips.update(ips_)

    print(f'MAC: {macs}, IP: {ips}')
    # get normal_durations
    normal_flows = device_meta['no_interaction'][0]
    normal_durations = get_durations(normal_flows)

    # get subflow_interval
    q_flow_dur = 0.9
    subflow_interval = np.quantile(normal_durations, q=q_flow_dur)  # median  of flow_durations
    print(f'---subflow_interval: ', subflow_interval, f', q_flow_dur: {q_flow_dur}')

    subflow_device_meta = {'q_flow_dur': q_flow_dur, 'subflow_interval': subflow_interval,
                           'normal_durations': normal_durations}
    for i, (activity_name, vs_) in enumerate(device_meta.items()):
        activity_flows, file_flows = vs_
        subflows = []
        for file_, flows_ in file_flows:
            subflows_ = flow2subflows(flows_, interval=subflow_interval, num_pkt_thresh=2, verbose=False)
            subflows += subflows_
        print(i, activity_name, len(activity_flows), len(subflows))
        subflow_device_meta[activity_name] = subflows[:]

    print('\n')
    # print subflow results
    for i, (key, vs_) in enumerate(sorted(subflow_device_meta.items())):
        if type(vs_) == list:
            print(i, key, len(vs_))
        else:
            print(i, key, vs_)


if __name__ == '__main__':
    for device in ['refrigerator', 'nestcam', 'alexa']:  # ['refrigerator', 'nestcam', 'alexa']
        print(f'\n***device: {device}')
        main(device)
