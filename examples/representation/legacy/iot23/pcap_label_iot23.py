"""Preprocess iot23 raw pcap and label.txt, such as filter ip

    download:
    wget -P CTU-IoT-Malware-Capture-7-1  https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-7-1/bro/conn.log.labeled --no-check-certificate &

"""
import os.path as pth

from itod_kjl.data.pcap.IoT23 import filter_pcap, filter_label, txt2csv_IoT23


def process(root_dir, data_name):
    if data_name == "CTU-IoT-Malware-Capture-1-1":
        ip = '192.168.100.103'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-05-09-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-3-1":
        ip = '192.168.2.5'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, '2018-05-21_capture.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-5-1":
        ip = '192.168.2.3'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, '2018-09-21-capture.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

        # data 2
        ip = '192.168.1.132'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'CTU-Honeypot-Capture-4-1/2018-10-25-14-06-32-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'CTU-Honeypot-Capture-4-1/conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-7-1":
        ip = '192.168.100.108'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-07-20-17-31-20-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-8-1":
        ip = '192.168.100.113'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-07-31-15-15-09-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)
    elif data_name == "CTU-IoT-Malware-Capture-9-1":
        ip = '192.168.100.111'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-07-25-10-53-16-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == 'CTU-IoT-Malware-Capture-20-1':
        ip = '192.168.100.103'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-10-02-13-12-30-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-33-1":
        # data 2
        ip = '192.168.1.197'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2018-12-20-21-10-00-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-48-1":
        # data 2
        ip = '192.168.1.200'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2019-02-28-19-15-13-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)

    elif data_name == "CTU-IoT-Malware-Capture-49-1":
        # data 2
        ip = '192.168.1.193'
        src = True
        # # filter ip in pcap
        pcap = pth.join(root_dir, data_name, f'2019-02-28-20-50-15-{ip}.pcap')
        file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
        file_out = filter_pcap(pcap, ip, src, file_out)
        print(file_out)

        # filter ip in label txt
        label_file = pth.join(root_dir, data_name, 'conn.log.labeled.txt')
        file_out = pth.join(root_dir, data_name, f'conn.log.labeled.csv')
        label_file = txt2csv_IoT23(label_file, file_out)
        file_out = pth.join(root_dir, data_name, f'src_{ip}.csv')
        file_out = filter_label(label_file, ip, src, file_out)
        print(file_out)


def main():
    root_dir = 'data/data_iot23/pcaps'

    datasets = [

        # # "CTU-IoT-Malware-Capture-1-1",
        # # "CTU-IoT-Malware-Capture-3-1",
        # # 'CTU-Honeypot-Capture-5-1'  # two normals
        "CTU-IoT-Malware-Capture-7-1",
        # # "CTU-IoT-Malware-Capture-8-1",
        # # 'CTU-IoT-Malware-Capture-9-1', # costs too much time
        # 'CTU-IoT-Malware-Capture-20-1',
        # # 'CTU-IoT-Malware-Capture-33-1', # costs too much time
        # 'CTU-IoT-Malware-Capture-48-1',
        # 'CTU-IoT-Malware-Capture-49-1',
    ]

    for data_name in datasets:
        print(f'*** {data_name}')
        process(root_dir, data_name)


def demo():
    ip = '192.168.100.108'
    src = True
    # # # filter ip in pcap
    # pcap = pth.join(root_dir, data_name, f'2018-07-20-17-31-20-{ip}.pcap')
    # file_out = pth.join(root_dir, data_name, f'src_{ip}.pcap')
    # file_out = filter_pcap(pcap, ip, src, file_out)
    # print(file_out)

    # # filter ip in label txt
    label_file = 'data/data_reprst/pcaps/CTU_IOT23/CTU-IoT-Malware-Capture-7-1/CTU-IoT-Malware-Capture-7-1-conn.log.labeled.txt'
    file_out = label_file + '.csv'
    label_file = txt2csv_IoT23(label_file, file_out)
    file_out = label_file + f'-src_{ip}.csv'
    file_out = filter_label(label_file, ip, src, file_out)
    print(file_out)


if __name__ == '__main__':
    # main()
    demo()
