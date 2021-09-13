"""Data process interface

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

#############################################################################################
import shutil
from shutil import copyfile

# 3. local libraries
from itod.pparser.dataset import *
from itod.pparser.pcap import *


class DataFactory:
    """Main entrance for data process, i.e., interface

    """

    def __init__(self, dataset_name='', params={}):
        """

        Parameters
        ----------
        dataset_name: str
            UNB, CTU, etc.

        params: dict
            stored experimental params, which will be updated in DataFactory
        """
        self.dataset_name = dataset_name
        self.params = params

        self.params['dataset_name'] = dataset_name
        self.verbose = self.params['verbose']

    @func_notation
    def run(self):
        """ Get all features from pcap

        Returns
        -------

        """
        if 'UNB/CICIDS_2017' in self.dataset_name:
            self.dataset_inst = DS10_UNB_IDS(dataset_name=self.dataset_name, params=self.params)
        elif 'UCHI/IOT_2019/smtv' in self.dataset_name:
            self.dataset_inst = DS20_PU_SMTV(dataset_name=self.dataset_name, params=self.params)
        elif 'DS30_OCS_IoT' in self.dataset_name:
            self.dataset_inst = DS30_OCS_IoT(dataset_name=self.dataset_name, params=self.params)
        elif 'CTU/IOT_2017/' in self.dataset_name:
            self.dataset_inst = DS40_CTU_IoT(dataset_name=self.dataset_name, params=self.params)
        elif 'MAWI/WIDE_2019' in self.dataset_name or 'MAWI/WIDE_2020' in self.dataset_name:
            self.dataset_inst = DS50_MAWI_WIDE(dataset_name=self.dataset_name, params=self.params)
        elif 'UCHI/IOT_2019' in self.dataset_name:
            self.dataset_inst = DS60_UChi_IoT(dataset_name=self.dataset_name, params=self.params)
        elif self.dataset_name.startswith('DEMO_'):
            if self.dataset_name.startswith('DEMO_IDS'):
                self.dataset_inst = DS10_UNB_IDS(dataset_name=self.dataset_name,
                                                 params=self.params)  # test DS10_UNB_IDS
            elif self.dataset_name.startswith('DEMO_SMTV'):
                self.dataset_inst = DS20_PU_SMTV(dataset_name=self.dataset_name,
                                                 params=self.params)  # test DS20_PU_SMTV
        else:
            msg = f'{self.dataset_name} is not correct.'
            raise ValueError(msg)

        #############################################################################################
        # 1) pcap2features, 2) get train set and test set 3) update params
        self.dataset_inst.run()
        # dataset_dict={'iat_dict':{}, 'fft_dict':...}, stored all data
        pprint(self.dataset_inst.dataset_dict, name='DataFactory.dataset_inst.dataset_dict', verbose=self.verbose)

        #############################################################################################
        # ignore some dataset.
        x_train, y_train, x_test, y_test = self.dataset_inst.dataset_dict['iat_dict']['data']
        if len(y_train) < 100 or x_train.shape[-1] > 1000:
            msg = f'x_train.shape: {x_train.shape} is too small or feature dimension is too large'
            raise ValueError(msg)


class DS10_UNB_IDS(PCAP, Dataset):
    """ Actual class used to process data

    """

    def __init__(self, dataset_name='', params={}):
        super(DS10_UNB_IDS, self).__init__()
        self.dataset_name = dataset_name
        self.params = params

        self.verbose = self.params['verbose']
        self.params['dataset_name'] = dataset_name

        # subflow: boolean (True or False)
        # if spit flow or not
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']

        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):

        self.label = ''.upper()
        if self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.5':
            self.srcIP = '192.168.10.5'
        elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.8':
            self.srcIP = '192.168.10.8'
        elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.9':
            self.srcIP = '192.168.10.9'
        elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.14':
            self.srcIP = '192.168.10.14'
        elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.15':
            self.srcIP = '192.168.10.15'
        # TUESDAY
        elif self.dataset_name == 'DS10_UNB_IDS/Tuesday/Tuesday-srcIP_172.16.0.1':  # one IP might have multi attacks, so we use IP instead of attack label here.
            self.srcIP = '172.16.0.1'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Tuesday/Tuesday-srcIP_172.16.0.1/FTP-PATATOR':
            self.srcIP = '172.16.0.1-FTP-PATATOR'
            ANOMALY = 'FTP-PATATOR'
        elif self.dataset_name == 'DS10_UNB_IDS/Tuesday/Tuesday-srcIP_172.16.0.1/SSH-PATATOR':
            self.srcIP = '172.16.0.1-SSH-PATATOR'
            ANOMALY = 'SSH-PATATOR'
        # WEDNESDAY
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1':
            self.srcIP = '172.16.0.1'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1/DOS-SLOWLORIS':
            self.srcIP = '172.16.0.1-DOS-SLOWLORIS'
            ANOMALY = 'DOS-SLOWLORIS'
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1/DOS-SLOWHTTPTEST':
            self.srcIP = '172.16.0.1-DOS-SLOWHTTPTEST'
            ANOMALY = 'DOS-SLOWHTTPTEST'
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1/DOS-HULK':
            self.srcIP = '172.16.0.1-DOS-HULK'
            ANOMALY = 'DOS-HULK'
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1/DOS-GOLDENEYE':
            self.srcIP = '172.16.0.1-DOS-GOLDENEYE'
            ANOMALY = 'DOS-GOLDENEYE'
        elif self.dataset_name == 'DS10_UNB_IDS/Wednesday/Wednesday-srcIP_172.16.0.1/HEARTBLEED':
            self.srcIP = '172.16.0.1-HEARTBLEED'
            ANOMALY = 'HEARTBLEED'
        # Thursday
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_172.16.0.1':
            self.srcIP = '172.16.0.1'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_172.16.0.1/WEB-ATTACK-BRUTE-FORCE':
            self.srcIP = '172.16.0.1-WEB-ATTACK-BRUTE-FORCE'
            ANOMALY = 'WEB-ATTACK-BRUTE-FORCE'
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_172.16.0.1/WEB-ATTACK-XSS':
            self.srcIP = '172.16.0.1-WEB-ATTACK-XSS'
            ANOMALY = 'WEB-ATTACK-XSS'
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_172.16.0.1/WEB-ATTACK-SQL-INJECTION':
            self.srcIP = '172.16.0.1-WEB-ATTACK-SQL-INJECTION'
            ANOMALY = 'WEB-ATTACK-SQL-INJECTION'
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_192.168.10.8':
            self.srcIP = '192.168.10.8'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Thursday/Thursday-srcIP_192.168.10.8/INFILTRATION':
            self.srcIP = '192.168.10.8-INFILTRATION'
            ANOMALY = 'INFILTRATION'
        # Friday
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.5':
            self.srcIP = '192.168.10.5'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.5/BOT':
            self.srcIP = '192.168.10.5-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.8':
            self.srcIP = '192.168.10.8'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.8/BOT':
            self.srcIP = '192.168.10.8-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.9':
            self.srcIP = '192.168.10.9'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.9/BOT':
            self.srcIP = '192.168.10.9-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.12':
            self.srcIP = '192.168.10.12'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.12/BOT':
            self.srcIP = '192.168.10.12-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.14':
            self.srcIP = '192.168.10.14'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.14/BOT':
            self.srcIP = '192.168.10.14-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.15':
            self.srcIP = '192.168.10.15'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.15/BOT':
            self.srcIP = '192.168.10.15-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.17':
            self.srcIP = '192.168.10.17'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.17/BOT':
            self.srcIP = '192.168.10.17-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_205.174.165.73':
            self.srcIP = '205.174.165.73'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_205.174.165.73/BOT':
            self.srcIP = '205.174.165.73-BOT'
            ANOMALY = 'BOT'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_172.16.0.1':
            self.srcIP = '172.16.0.1'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_172.16.0.1/PORTSCAN':
            self.srcIP = '172.16.0.1-PORTSCAN'
            ANOMALY = 'PORTSCAN'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_172.16.0.1/DDOS':
            self.srcIP = '172.16.0.1-DDOS'
            ANOMALY = 'DDOS'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.50':
            self.srcIP = '192.168.10.50'
            ANOMALY = 'ALL_ANOMALY'
        elif self.dataset_name == 'DS10_UNB_IDS/Friday/Friday-srcIP_192.168.10.50/DDOS':
            self.srcIP = '192.168.10.50-DDOS'
            ANOMALY = 'DDOS'

        elif self.dataset_name == 'DEMO_IDS/DS-srcIP_192.168.10.5':
            self.srcIP = '192.168.10.5'
        else:
            raise ValueError('dataset does not exist.')
        self.params['srcIP'] = self.srcIP  # update params

        if self.params['data_cat'] == 'INDV':  # data_catagory
            pass
            # if 'Tuesday' in self.dataset_name or 'Wednesday' in self.dataset_name or 'Thursday' in self.dataset_name \
            #         or 'Friday' in self.dataset_name:
            #     self.pcap_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
            #                                    data_cat='INDV', file_name=f'srcIP_{self.srcIP}.pcap')
            #     self.label_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
            #                                     data_cat='INDV', file_name=f'srcIP_{self.srcIP}.csv')
            # else:
            #     self.pcap_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
            #                                    data_cat='Friday-WorkingHours', file_name=f'srcIP_{self.srcIP}.pcap')
            #     self.label_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
            #                                     data_cat='Friday-WorkingHours', file_name=f'srcIP_{self.srcIP}.csv')

        elif self.params['data_cat'] == 'AGMT':
            if 'Tuesday' in self.dataset_name or 'Wednesday' in self.dataset_name or 'Thursday' in self.dataset_name \
                    or 'Friday' in self.dataset_name:
                pass
                # if ANOMALY != 'ALL_ANOMALY':
                #     dataset_name = os.path.dirname(self.dataset_name)
                # else:
                #     dataset_name = self.dataset_name
                #
                # self.pcap_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                data_cat=f'AGMT/{ANOMALY}', file_name=f'AGMT-srcIP_{self.srcIP}.pcap')
                # self.label_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                 data_cat=f'AGMT/{ANOMALY}', file_name=f'AGMT-srcIP_{self.srcIP}.csv')
                #
                # print(f'self.label_file: {self.label_file}', os.getcwd(), os.path.exists(self.label_file),
                #       os.path.exists(str(self.label_file)))
                # if self.params['overwrite']:
                #     if ANOMALY != 'ALL_ANOMALY':
                #         dataset_name = os.path.dirname(self.dataset_name)
                #     else:
                #         dataset_name = self.dataset_name
                #     if 'srcIP_172.16.0.1' in self.dataset_name:
                #         normal_srcIP = '192.168.10.51'
                #         pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat='INDV', file_name=f'srcIP_172.16.0.1.pcap')
                #         pcap_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat=f'Monday-srcIP_{normal_srcIP}',
                #                                     file_name=f'srcIP_{normal_srcIP}.pcap')
                #         merge_pcaps([pcap_file_1, pcap_file_2], mrg_pcap_path=self.pcap_file)
                #
                #         label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat='INDV', file_name=f'{ANOMALY}/srcIP_172.16.0.1.csv')
                #         label_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat=f'Monday-srcIP_{normal_srcIP}',
                #                                      file_name=f'BENIGN/srcIP_{normal_srcIP}.csv')
                #         merge_labels([label_file_1, label_file_2], mrg_label_path=self.label_file)
                #     elif '/BOT' in self.dataset_name or '/DDOS' in self.dataset_name:
                #         dataset_name = os.path.dirname(self.dataset_name)
                #         srcIP = self.srcIP.split('-')[0]
                #         pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat='INDV', file_name=f'{ANOMALY}/srcIP_{srcIP}.pcap')
                #         pcap_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat=f'Monday-srcIP_{srcIP}',
                #                                     file_name=f'srcIP_{srcIP}.pcap')
                #         merge_pcaps([pcap_file_1, pcap_file_2], mrg_pcap_path=self.pcap_file)
                #
                #         label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat='INDV', file_name=f'{ANOMALY}/srcIP_{srcIP}.csv')
                #         label_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat=f'Monday-srcIP_{srcIP}',
                #                                      file_name=f'srcIP_{srcIP}.csv')
                #         merge_labels([label_file_1, label_file_2], mrg_label_path=self.label_file)
                #
                #     else:
                #         pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat='INDV', file_name=f'srcIP_{self.srcIP}.pcap')
                #         pcap_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                     data_cat=f'Monday-srcIP_{self.srcIP}',
                #                                     file_name=f'srcIP_{self.srcIP}.pcap')
                #         merge_pcaps([pcap_file_1, pcap_file_2], mrg_pcap_path=self.pcap_file)
                #
                #         label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat='INDV', file_name=f'srcIP_{self.srcIP}.csv')
                #         label_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=dataset_name,
                #                                      data_cat=f'Monday-srcIP_{self.srcIP}',
                #                                      file_name=f'srcIP_{self.srcIP}.csv')
                #         merge_labels([label_file_1, label_file_2], mrg_label_path=self.label_file)

            else:
                self.pcap_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                               data_cat='AGMT-WorkingHours', file_name=f'pc_{self.srcIP}_AGMT.pcap')
                self.label_file = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                data_cat='AGMT-WorkingHours', file_name=f'pc_{self.srcIP}_AGMT.csv')
                if not os.path.exists(os.path.dirname(self.pcap_file)):
                    os.makedirs(os.path.dirname(self.pcap_file))
                print(f'self.label_file: {os.path.abspath(self.label_file)}', os.getcwd(),
                      os.path.exists(self.label_file),
                      os.path.exists(str(self.label_file)))
                if not os.path.exists(self.pcap_file) or not os.path.exists(self.label_file) or self.params[
                    'overwrite']:
                    print('file does not exist!')

                    # # Friday morning dataset
                    friday_pacp = get_file_path(ipt_dir=self.params['original_ipt_dir'],
                                                dataset_name='UNB/CICIDS_2017/',
                                                data_cat='pcaps/Friday',
                                                file_name='Friday-WorkingHours.pcap')
                    # only use the selected subpcaps
                    friday_pacp = extract_subpcap(friday_pacp, out_file='', start_time='2017-07-07 08:00:00',
                                                  end_time='2017-07-07 13:00:00', verbose=20, keep_original=True)
                    friday_label = get_file_path(ipt_dir=self.params['original_ipt_dir'],
                                                 dataset_name='UNB/CICIDS_2017/',
                                                 data_cat='labels/Friday',
                                                 file_name='Friday-WorkingHours-Morning.pcap_ISCX.csv')
                    print(f'friday_pacp: {friday_pacp}, friday_label: {friday_label}')

                    if '192.168.10.5' in self.pcap_file or '192.168.10.8' in self.pcap_file or '192.168.10.9' in self.pcap_file \
                            or '192.168.10.14' in self.pcap_file:
                        # # Monday dataset
                        # monday_pcap = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='UNB/CICIDS_2017/',
                        #                           data_cat='pcaps/Monday',
                        #                           file_name='Monday-WorkingHours.pcap')
                        # # only use the selected subpcaps
                        # monday_pcap = extract_subpcap(monday_pcap, out_file='', start_time='2017-07-03 08:00:00',
                        #                             end_time='2017-07-03 10:00:00', verbose=20, keep_original=True)
                        # monday_label = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='UNB/CICIDS_2017/',
                        #                            data_cat='labels/Monday',
                        #                            file_name='Monday-WorkingHours.pcap_ISCX.csv')
                        # print(f'monday_pcap: {monday_pcap}, monday_label: {monday_label}')

                        pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                    data_cat='Friday-WorkingHours',
                                                    file_name=f'pc_{self.srcIP}.pcap')
                        # pcap_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                        #                             data_cat='Monday-WorkingHours',
                        #                             file_name=f'pc_{self.srcIP}.pcap')
                        pcap_file_1 = filter_ip(friday_pacp, out_file=pcap_file_1, ips=[self.srcIP],
                                                direction=self.params['direction'],
                                                keep_original=True)
                        # pcap_file_2 = filter_ip(monday_pcap, out_file=pcap_file_2, ips=[self.srcIP], direction =self.params['direction'],
                        #                         keep_original=True)
                        # merge_pcaps([pcap_file_1, pcap_file_2], mrg_pcap_path=self.pcap_file)
                        copyfile(pcap_file_1, self.pcap_file)

                        label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                     data_cat='Friday-WorkingHours',
                                                     file_name=f'pc_{self.srcIP}.csv')
                        # label_file_2 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                        #                              data_cat='Monday-WorkingHours',
                        #                              file_name=f'pc_{self.srcIP}.csv')
                        label_file_1 = filter_csv_ip(friday_label, out_file=label_file_1, ips=[self.srcIP],
                                                     direction=self.params['direction'], keep_original=True)
                        # label_file_2 = filter_csv_ip(monday_label, out_file=label_file_2, ips=[self.srcIP],
                        #                              direction =self.params['direction'], keep_original=True)
                        # merge_labels([label_file_1, label_file_2], mrg_label_path=self.label_file)
                        copyfile(label_file_1, self.label_file)
                    elif '192.168.10.15' in self.pcap_file:
                        # only use the selected subpcaps
                        friday_pacp = extract_subpcap(friday_pacp, out_file='', start_time='2017-07-07 08:30:00',
                                                      end_time='2017-07-07 13:00:00', verbose=20, keep_original=True)
                        pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                    data_cat='Friday-WorkingHours', file_name=f'pc_{self.srcIP}.pcap')
                        pcap_file_1 = filter_ip(friday_pacp, out_file=pcap_file_1, ips=[self.srcIP],
                                                direction=self.params['direction'],
                                                keep_original=True)
                        copyfile(pcap_file_1, self.pcap_file)

                        label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                     data_cat='Friday-WorkingHours', file_name=f'pc_{self.srcIP}.csv')
                        label_file_1 = filter_csv_ip(friday_label, out_file=label_file_1, ips=[self.srcIP],
                                                     direction=self.params['direction'], keep_original=True)
                        copyfile(label_file_1, self.label_file)

                    else:

                        pcap_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                    data_cat='Friday-WorkingHours', file_name=f'pc_{self.srcIP}.pcap')
                        pcap_file_1 = filter_ip(friday_pacp, out_file=pcap_file_1, ips=[self.srcIP],
                                                direction=self.params['direction'],
                                                keep_original=True)
                        copyfile(pcap_file_1, self.pcap_file)

                        label_file_1 = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                                     data_cat='Friday-WorkingHours', file_name=f'pc_{self.srcIP}.csv')
                        label_file_1 = filter_csv_ip(friday_label, out_file=label_file_1, ips=[self.srcIP],
                                                     direction=self.params['direction'], keep_original=True)
                        copyfile(label_file_1, self.label_file)

        elif self.params['data_cat'] == 'MIX':
            pass
            #
            # pcap_file_lst = []
            # label_file_lst = []
            # self.srcIP_lst = ['192.168.10.5', '192.168.10.8', '192.168.10.9', '192.168.10.14', '192.168.10.15']
            # for i, srcIP in enumerate(self.srcIP_lst):
            #     sub_data_name = f'DS1{i + 1}-srcIP_{self.srcIP}'
            #     pcap_file_i = get_file_path(ipt_dir=self.params['ipt_dir'],
            #                                 sub_dir=f'{sub_data_name}/Friday-WorkingHours',
            #                                 file_name=f'srcIP_{self.srcIP}.pcap')
            #     pcap_file_lst.append(pcap_file_i)
            #     label_file_i = get_file_path(ipt_dir=self.params['ipt_dir'],
            #                                  sub_dir=f'{sub_data_name}/Friday-WorkingHours',
            #                                  file_name=f'srcIP_{self.srcIP}.csv')
            #     label_file_lst.append(label_file_i)
            #
            # sub_data_name = 'DS16-five_srcIPs'
            # mrg_srcIP = 'five_srcIPs'
            # self.pcap_file = get_file_path(ipt_dir=self.params['ipt_dir'], sub_dir=f'{sub_data_name}/MIX-WorkingHours',
            #                                file_name=f'srcIP_{mrg_srcIP}.pcap')
            # self.label_file = get_file_path(ipt_dir=self.params['ipt_dir'], sub_dir=f'{sub_data_name}/MIX-WorkingHours',
            #                                 file_name=f'srcIP_{mrg_srcIP}.csv')
            # if self.params['overwrite']:
            #     merge_pcaps(pcap_file_lst, mrg_pcap_path=self.pcap_file)
            #     merge_labels(label_file_lst, mrg_label_path=self.label_file)

    @func_notation
    def get_flows_labels(self):
        ##############################################################################################
        # 1. get pcap_file and label_file
        self.get_files()

        ##############################################################################################
        # 2. get flows and labels
        output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
            self.subflow_interval) + '_q_flow_duration=' + str(self.params['q_flow_dur'])

        try:
            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 pcap_file to flows, label_file to labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcap_label(self.pcap_file, self.label_file,
                                                                          subflow=self.params['subflow'],
                                                                          output_flows_labels=output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.pcap_file)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)

        except Exception as e:
            print(f'Error: {e}, occurs in {os.path.relpath(self.get_flows_labels.__code__.co_filename)} at '
                  f'{self.get_flows_labels.__code__.co_firstlineno}')

        return self.flows, self.labels


class DS20_PU_SMTV(PCAP, Dataset):

    def __init__(self, dataset_name='', params={}):
        super(DS20_PU_SMTV, self).__init__()
        self.dataset_name = dataset_name
        self.params = params

        self.params['dataset_name'] = dataset_name
        self.verbose = self.params['verbose']

        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']

        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        """ Get pcap_file and label_file

        Returns
        -------

        """
        if self.params['data_cat'] == 'INDV':  # only Friday data
            if self.dataset_name == 'UCHI/IOT_2019/smtv_10.42.0.1':
                self.srcIP = '10.42.0.1'
                self.params['srcIP'] = self.srcIP

                # filter pcap
                file_name = 'pc_10.42.0.1_normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                               file_name=file_name)
                copy_file(pcap_file, self.nrml_pcap)

                file_name = 'pc_10.42.0.119_anomaly.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                               file_name=file_name)
                copy_file(pcap_file, self.anml_pcap)
        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        """ Get flows and labels

        Returns
        -------

        """
        ##############################################################################################
        # 1. get pcap_file and label_file
        self.get_files()

        ##############################################################################################
        # 2. get flows and labels
        self.pcap_file = os.path.join(os.path.dirname(self.nrml_pcap)) + '/'
        output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
            self.subflow_interval) + '_q_flow_duration=' + str(self.params['q_flow_dur'])
        try:
            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 get flows and labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcaps(pcap_file_lst=[self.nrml_pcap, self.anml_pcap],
                                                                     subflow=self.params['subflow'],
                                                                     output_flows_labels=output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.anml_pcap)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels

    # @func_notation
    # def _get_nrml_anml_smart_tv_data(self, ipt_dir, keep_ip='10.42.0.119', overwrite=True):
    #     pcap_file_lst = []
    #     nrml_dir = os.path.join(ipt_dir, 'DS31-srcIP_192.168.0.13', 'normal')
    #     srcIP = '10.42.0.1'
    #     for i, pcap in enumerate(os.listdir(nrml_dir)):
    #         if not pcap.endswith('.pcap'):
    #             print(f'i:{i + 1}, pcap:{pcap}')
    #             continue
    #         pcap_file_lst.append(os.path.join(nrml_dir, pcap))
    #     sub_data_name = 'DS21-srcIP_10.42.0.1'
    #     nrml_pcap = get_file_path(ipt_dir, sub_dir=f'{sub_data_name}/pcaps', file_name='normal.pcap')
    #     merge_pcaps(pcap_file_lst, mrg_pcap_path=nrml_pcap)
    #     src_nrml_pcap = get_file_path(ipt_dir, sub_dir=f'{sub_data_name}/pcaps',
    #                                   file_name=f'srcIP_{srcIP}_normal.pcap')
    #     # if self.overwrite:
    #     #     keep_ips_in_pcap(input_file=nrml_pcap, output_file=src_nrml_pcap, kept_ips=[srcIP])
    #
    #     pcap_file_lst = []
    #     srcIP = '10.42.0.119'
    #     anml_dir = os.path.join(ipt_dir, 'DS31-srcIP_192.168.0.13', 'anomaly')
    #     for i, pcap in enumerate(os.listdir(anml_dir)):
    #         if not pcap.endswith('.pcap'):
    #             print(f'i:{i + 1}, pcap:{pcap}')
    #             continue
    #         pcap_file_lst.append(os.path.join(anml_dir, pcap))
    #     anml_pcap = get_file_path(ipt_dir, sub_dir=f'{sub_data_name}/pcaps', file_name='anomaly.pcap')
    #     merge_pcaps(pcap_file_lst, mrg_pcap_path=anml_pcap)
    #     src_anml_pcap = get_file_path(ipt_dir, sub_dir=f'{sub_data_name}/pcaps',
    #                                   file_name=f'srcIP_{srcIP}_anomaly.pcap')
    #     # if overwrite:
    #     #     keep_ips_in_pcap(input_file=anml_pcap, output_file=src_anml_pcap, kept_ips=[srcIP])
    #
    #     return 'multi-srcIPs', src_nrml_pcap, src_anml_pcap


class DS30_OCS_IoT(PCAP, Dataset):
    def __init__(self, dataset_name='', params={}):
        super(DS30_OCS_IoT, self).__init__()
        self.dataset_name = dataset_name
        self.params = params
        self.dataset_dict = OrderedDict()
        self.params['dataset_dict'] = self.dataset_dict
        self.verbose = self.params['verbose']
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']
        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        """

        Returns
        -------

        """
        if self.params['data_cat'] == 'INDV':  # only Friday data
            if self.dataset_name == 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13':
                self.srcIP = '192.168.0.13'

                # filter pcap
                file_name = 'MIRAI/benign-dec-EZVIZ-ip_src-192.168.0.13-normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          )
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'], dataset_name=self.dataset_name,
                                               file_name=file_name)
                copy_file(pcap_file, self.nrml_pcap)

                file_name = 'MIRAI/mirai-udpflooding-1-dec-EZVIZ-ip_src-192.168.0.13-anomaly.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(self.params['ipt_dir'], dataset_name=self.dataset_name,
                                               file_name=file_name)
                copy_file(pcap_file, self.anml_pcap)

                self.params['srcIP'] = self.srcIP
            else:
                pass

        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            # ipt_dir = '-input_data/smart-tv-roku-data'
            msg = f'MIX doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        """ Get flows and labels

        Returns
        -------

        """
        self.get_files()

        try:
            self.pcap_file = os.path.join(os.path.dirname(self.nrml_pcap)) + '/'
            output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
                self.subflow_interval) + '_q_flow_duration=' + str(
                self.params['q_flow_dur'])

            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')
                    # os.remove(output_flows_labels)

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcaps(pcap_file_lst=[self.nrml_pcap, self.anml_pcap],
                                                                     subflow=self.params['subflow'],
                                                                     output_flows_labels=output_flows_labels)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels


class DS40_CTU_IoT(PCAP, Dataset):
    def __init__(self, dataset_name='', params={}):
        super(DS40_CTU_IoT, self).__init__()
        self.dataset_name = dataset_name
        self.params = params
        self.dataset_dict = OrderedDict()
        self.params['dataset_dict'] = self.dataset_dict
        self.verbose = self.params['verbose']
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']
        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        """ Get pcap_file and label_file

        Returns
        -------

        """
        if self.params['data_cat'] == 'INDV':  # only Friday data
            """
            https://www.stratosphereips.org/datasets-iot
            Malware on IoT Dataset
            """
            if self.dataset_name == 'CTU/IOT_2017/pc_10.0.2.15':
                self.srcIP = '10.0.2.15'
                self.params['srcIP'] = self.srcIP
                # filter pcap
                # file_name = '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
                file_name = 'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='CTU/IOT_2017',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=['192.168.1.196'], direction=self.params['direction'])

                # file_name = '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap'
                file_name = 'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='CTU/IOT_2017',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-192.168.1.195_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['192.168.1.195'], direction=self.params['direction'])
            else:
                pass

        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            # ipt_dir = '-input_data/smart-tv-roku-data'
            msg = f'MIX doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        ##############################################################################################
        # 1. get flows and labels
        self.get_files()

        try:
            self.pcap_file = os.path.join(os.path.dirname(self.nrml_pcap)) + '/'
            output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
                self.subflow_interval) + '_q_flow_duration=' + str(
                self.params['q_flow_dur'])

            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')
                    # os.remove(output_flows_labels)

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 get flows and labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcaps(pcap_file_lst=[self.nrml_pcap, self.anml_pcap],
                                                                     subflow=self.params['subflow'],
                                                                     output_flows_labels=output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.anml_pcap)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels


class DS50_MAWI_WIDE(PCAP, Dataset):
    def __init__(self, dataset_name='', params={}):
        super(DS50_MAWI_WIDE, self).__init__()
        self.dataset_name = dataset_name
        self.params = params
        self.dataset_dict = OrderedDict()
        self.params['dataset_dict'] = self.dataset_dict
        self.verbose = self.params['verbose']
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']
        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        """ Get pcap_file and label_file

        Returns
        -------

        """
        if self.params['data_cat'] == 'INDV':
            if self.dataset_name == 'MAWI/WIDE_2019/pc_202.171.168.50':
                # "http://mawi.wide.ad.jp/mawi/samplepoint-F/2019/201912071400.html"
                self.srcIP = '202.171.168.50'
                # self.nrml_pcap = self.params[
                #                      'ipt_dir'] + '/MAWI/WIDE_2019/pc_202.171.168.50/201912071400-10000000pkts_00000_src_202_171_168_50_normal.pcap'
                # self.anml_pcap = self.params[
                #                      'ipt_dir'] + '/MAWI/WIDE_2019/pc_202.171.168.50/201912071400-10000000pkts_00000_src_202_4_27_109_anomaly.pcap'
                # self.params['srcIP'] = self.srcIP

                # filter pcap
                # file_name = 'samplepoint-F_201912071400.pcap'
                # editcap -c 300000000 in_file out_file
                # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
                if self.params['direction'] == 'src_dst':
                    file_name = 'samplepoint-F_201912071400-src_dst_202.171.168.50-5000000.pcap'
                else:
                    file_name = 'samplepoint-F_201912071400-src_dst_202.171.168.50.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2019',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                file_name = 'samplepoint-F_201912071400-src_dst_202.4.27.109.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2019',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-202.4.27.109_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['202.4.27.109'], direction=self.params['direction'])
            elif self.dataset_name == 'MAWI/WIDE_2020/pc_203.78.7.165':
                # "http://mawi.wide.ad.jp/mawi/samplepoint-F/2020/202007011400.html"
                self.srcIP = '203.78.7.165'
                # filter pcap
                # file_name = 'samplepoint-F_201912071400.pcap'
                # editcap -c 300000000 in_file out_file
                # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
                file_name = 'samplepoint-F_202007011400-src_dst_203.78.7.165.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                file_name = 'samplepoint-F_202007011400-src_dst_185.8.54.240.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-185.8.54.240_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['185.8.54.240'], direction=self.params['direction'])
            elif self.dataset_name == 'MAWI/WIDE_2020/pc_203.78.4.32':
                # "http://mawi.wide.ad.jp/mawi/samplepoint-F/2020/202007011400.html"
                self.srcIP = '203.78.4.32'
                # file_name = 'samplepoint-F_201912071400.pcap'
                # editcap -c 300000000 in_file out_file
                # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
                file_name = 'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # tshark -r samplepoint-F_202007011400.pcap -w samplepoint-F_202007011400-src_dst_163.191.181.125.pcap ip.addr=='163.191.181.125'
                file_name = 'samplepoint-F_202007011400.pcap-src_dst_202.75.33.114.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-202.75.33.114_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['202.75.33.114'], direction=self.params['direction'])

            elif self.dataset_name == 'MAWI/WIDE_2020/pc_203.78.4.32-2':
                # "http://mawi.wide.ad.jp/mawi/samplepoint-F/2020/202007011400.html"
                self.srcIP = '203.78.4.32'
                # file_name = 'samplepoint-F_201912071400.pcap'
                # editcap -c 300000000 in_file out_file
                # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
                file_name = 'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # tshark -r samplepoint-F_202007011400.pcap -w samplepoint-F_202007011400-src_dst_163.191.181.125.pcap ip.addr=='163.191.181.125'
                file_name = 'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-203.78.8.151_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['203.78.8.151'], direction=self.params['direction'])

            elif self.dataset_name == 'MAWI/WIDE_2020/pc_203.78.7.165-2':
                # "http://mawi.wide.ad.jp/mawi/samplepoint-F/2020/202007011400.html"
                self.srcIP = '203.78.7.165'
                # file_name = 'samplepoint-F_201912071400.pcap'
                # editcap -c 300000000 in_file out_file
                # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
                file_name = 'samplepoint-F_202007011400-src_dst_203.78.7.165.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}_normal.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # tshark -r samplepoint-F_202007011400.pcap -w samplepoint-F_202007011400-src_dst_163.191.181.125.pcap ip.addr=='163.191.181.125'
                file_name = 'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name='MAWI/WIDE_2020',
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-203.78.8.151_abnormal.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=['203.78.8.151'], direction=self.params['direction'])
        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            # ipt_dir = '-input_data/smart-tv-roku-data'
            msg = f'MIX doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        """ Get flows and labels

        Returns
        -------

        """
        ##############################################################################################
        # 2.1 get pcap_file and label_file
        self.get_files()

        try:
            self.pcap_file = os.path.join(os.path.dirname(self.nrml_pcap)) + '/'
            output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
                self.subflow_interval) + '_q_flow_duration=' + str(
                self.params['q_flow_dur'])

            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')
                    # os.remove(output_flows_labels)

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 get flows and labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcaps(pcap_file_lst=[self.nrml_pcap, self.anml_pcap],
                                                                     subflow=self.params['subflow'],
                                                                     output_flows_labels=output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.anml_pcap)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels


class DS60_UChi_IoT(PCAP, Dataset):
    def __init__(self, dataset_name='', params={}):
        super(DS60_UChi_IoT, self).__init__()
        """
        process the pcap:
        1) for each device, only extract the traffic generated by the device (i.e., only forward traffic, 
            ip.src==the deceive src IP) from fridge_cam_sound_ghome_2daysactiv.pcap
        2) separate the extracted traffic to two parts: without activity (idle: label as normal) and with activity 
            (label as anomaly), accroding to the timestamps by wireshark
        3) use normal to train, then evaluate the detection model on test set (normal + anomaly)


        wireshark process:
        (frame.time >= "Oct 15, 2012 16:00:00") && (frame.time <= "Oct 15, 2012 17:00:00")

        google home (in EST time):google_home-2daysactiv-src_192.168.143.20.pcap
            Normal: frame.time <= "Dec 13, 2019 13:30:00"
            Anomaly: (frame.time > "Dec 13, 2019 13:30:00") && (frame.time <= "Dec 13, 2019 13:37:00")

        samsung camera (in EST time): samsung_camera-2daysactiv-src_192.168.143.42.pcap
            Normal: frame.time <= "Dec 13, 2019 13:29:00"
            Anomaly: (frame.time > "Dec 13, 2019 13:30:00") && (frame.time <= "Dec 13, 2019 13:37:00")


        samsung fridge(in EST time):samsung_fridge-2daysactiv-src_192.168.143.43-normal.pcap
            Normal: frame.time <= "Dec 13, 2019 13:25:00"
            Anomaly: (frame.time > "Dec 13, 2019 13:25:00") && (frame.time <= "Dec 13, 2019 13:37:00")


        bose soundtouch(in EST time):bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap
            Normal: frame.time <= "Dec 13, 2019 13:24:00"
            Anomaly: (frame.time > "Dec 13, 2019 13:24:00") && (frame.time <= "Dec 13, 2019 13:37:00")

        """

        self.params = params
        # self.params['srcIP_lst'] = srcIP_lst
        self.dataset_dict = OrderedDict()
        self.params['dataset_dict'] = self.dataset_dict
        self.verbose = self.params['verbose']
        self.dataset_name = self.params['dataset_name']
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']
        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        if self.params['data_cat'] == 'INDV':  #
            if self.dataset_name == 'UCHI/IOT_2019/ghome_192.168.143.20':
                self.srcIP = '192.168.143.20'

                # filter pcap
                # file_name = 'google_home-2daysactiv-src_192.168.143.20-normal.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-ghome_normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # file_name = 'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-ghome_abnormal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=[self.srcIP], direction=self.params['direction'])

            elif self.dataset_name == 'UCHI/IOT_2019/scam_192.168.143.42':
                self.srcIP = '192.168.143.42'
                # filter pcap
                # file_name = 'samsung_camera-2daysactiv-src_192.168.143.42-normal.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-scam_normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # file_name = 'samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pca'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-scam_abnormal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=[self.srcIP], direction=self.params['direction'])

            elif self.dataset_name == 'UCHI/IOT_2019/sfrig_192.168.143.43':
                self.srcIP = '192.168.143.43'
                # filter pcap
                # file_name = 'samsung_fridge-2daysactiv-src_192.168.143.43-normal.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-sfrig_normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # file_name = 'samsung_fridge-2daysactiv-src_192.168.143.43-anomaly.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-sfrig_abnormal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=[self.srcIP], direction=self.params['direction'])

            elif self.dataset_name == 'UCHI/IOT_2019/bstch_192.168.143.48':
                self.srcIP = '192.168.143.48'

                # filter pcap
                # file_name = 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-bstch_normal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.nrml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.nrml_pcap, ips=[self.srcIP], direction=self.params['direction'])

                # file_name = 'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap'
                file_name = 'fridge_cam_sound_ghome_2daysactiv-bstch_abnormal.pcap'
                pcap_file = get_file_path(ipt_dir=self.params['original_ipt_dir'], dataset_name=self.dataset_name,
                                          file_name=file_name)
                self.anml_pcap = get_file_path(ipt_dir=self.params['ipt_dir'],
                                               dataset_name=self.dataset_name,
                                               file_name=file_name + f'-{self.srcIP}.pcap')
                filter_ip(pcap_file, self.anml_pcap, ips=[self.srcIP], direction=self.params['direction'])
            else:
                self.nrml_pcap = self.params[
                                     'ipt_dir'] + '/DS60_UChi_IoT/DS64-srcIP_192.168.143.48/bose_soundtouch_30-sample-src_192.168.143.48_normal.pcap'
                self.anml_pcap = self.params[
                                     'ipt_dir'] + '/DS60_UChi_IoT/DS64-srcIP_192.168.143.48/google_home-sample-src_192.168.143.20_anomaly.pcap'
                self.anml_pcap = self.params[
                                     'ipt_dir'] + '/DS60_UChi_IoT/DS64-srcIP_192.168.143.48/samsung_fridge-src_192.168.143.43_anomaly.pcap'
                self.srcIP = '192.168.143.48'
            self.params['srcIP'] = self.srcIP
        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            # ipt_dir = '-input_data/smart-tv-roku-data'
            msg = f'MIX doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        """

        Returns
        -------

        """
        ##############################################################################################
        # 1. get pcap_file and label_file
        self.get_files()

        try:
            self.pcap_file = os.path.join(os.path.dirname(self.nrml_pcap)) + '/'
            output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
                self.subflow_interval) + '_q_flow_duration=' + str(
                self.params['q_flow_dur'])

            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')

            if os.path.exists(output_flows_labels):
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 get flows and labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcaps(pcap_file_lst=[self.nrml_pcap, self.anml_pcap],
                                                                     subflow=self.params['subflow'],
                                                                     output_flows_labels=output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.anml_pcap)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels


class DS70_CTU_IoT(PCAP, Dataset):
    def __init__(self, dataset_name='', params={}):
        super(DS70_CTU_IoT, self).__init__()
        self.dataset_name = dataset_name
        self.params = params
        self.dataset_dict = OrderedDict()
        self.params['dataset_dict'] = self.dataset_dict
        self.verbose = self.params['verbose']
        self.subflow = self.params['subflow']
        self.subflow_interval = self.params['subflow_interval']
        self.overwrite = self.params['overwrite']

    @func_notation
    def get_files(self):
        if self.params['data_cat'] == 'INDV':  # only Friday data
            if self.dataset_name == 'DS70_CTU_IoT/DS71-srcIP_192.168.100.113':
                self.srcIP = '192.168.100.113'
                self.pcap_file = self.params[
                                     'ipt_dir'] + '/DS70_CTU_IoT/DS71-srcIP_192.168.100.113/2018-07-31-15-15-09-192.168.100.113.pcap'

                self.label_file = self.params[
                                      'ipt_dir'] + '/DS70_CTU_IoT/DS71-srcIP_192.168.100.113/192.168.100.113-conn.log.labeled-xlsx.csv'
                self.params['srcIP'] = self.srcIP

                if not os.path.exists(self.label_file):
                    input_file = self.params[
                                     'ipt_dir'] + '/DS70_CTU_IoT/DS71-srcIP_192.168.100.113/192.168.100.113-conn.log.labeled.txt'
                    conn_log_2_csv(input_file, output_file=self.label_file)


            elif self.dataset_name == 'DS70_CTU_IoT/DS71-srcIP_192.168.1.198':
                self.srcIP = '192.168.1.198'
                self.pcap_file = self.params[
                                     'ipt_dir'] + '/DS70_CTU_IoT/DS72-srcIP_192.168.1.198/2019-01-10-19-22-51-192.168.1.198_100000pkts.pcap'

                self.label_file = self.params[
                                      'ipt_dir'] + '/DS70_CTU_IoT/DS72-srcIP_192.168.1.198/2019-01-10-19-22-51-192.168.1.198.pcap-log.labeled.txt_reduce.txt_csv.txt'
                self.params['srcIP'] = self.srcIP

                if not os.path.exists(self.label_file):
                    input_file = self.params[
                                     'ipt_dir'] + '/DS70_CTU_IoT/DS72-srcIP_192.168.1.198/2019-01-10-19-22-51-192.168.1.198.pcap-log.labeled.txt'
                    conn_log_2_csv(input_file, output_file=self.label_file)

            else:
                pass


        elif self.params['data_cat'] == 'AGMT':  # Friday + Monday data
            msg = f'AGMT doesn\'t not be implemented yet.'
            raise ValueError(msg)
        elif self.params['data_cat'] == 'MIX':  # combine Friday data
            # ipt_dir = '-input_data/smart-tv-roku-data'
            msg = f'MIX doesn\'t not be implemented yet.'
            raise ValueError(msg)
        else:
            data_cat = self.params['data_cat']
            msg = f'{data_cat} is not correct.'
            raise ValueError(msg)

    @func_notation
    def get_flows_labels(self):
        """

        Returns
        -------

        """
        self.get_files()  # get pcap_file and label_file

        output_flows_labels = self.pcap_file + '-subflow_interval=' + str(
            self.subflow_interval) + '_q_flow_duration=' + str(
            self.params['q_flow_dur'])

        try:
            if self.overwrite:
                if os.path.exists(output_flows_labels):
                    suffex = time.strftime("%Y-%m-%d %H", time.localtime())
                    shutil.move(output_flows_labels, output_flows_labels + f'-{suffex}')
                    # os.remove(output_flows_labels)

            if os.path.exists(output_flows_labels):
                # with open(output_flows_labels, 'rb') as in_hdl:
                #     self.flows, self.labels, self.subflow_interval = pickle.load(in_hdl)
                self.flows, self.labels, self.subflow_interval = load_flows_label_pickle(output_flows_labels)
                self.params['subflow_interval'] = self.subflow_interval
            else:
                ##############################################################################################
                # 2.1 get flows and labels
                print(f'{output_flows_labels} does not exist.')
                self.flows, self.labels = self.pcap2flows_with_pcap_label(self.pcap_file, self.label_file,
                                                                          subflow=self.params['subflow'],
                                                                          output_flows_labels=output_flows_labels,
                                                                          label_file_type='CTU-IoT-23')
                if not os.path.exists(output_flows_labels + '-all.dat'):
                    # dump all flows, labels
                    dump_data((self.flows, self.labels, self.subflow_interval),
                              output_file=output_flows_labels + '-all.dat',
                              verbose=True)
                ##############################################################################################
                # 2.2 sample a part of flows and labels to do the following experiment
                self.flows, self.labels = random_select_flows(self.flows, self.labels,
                                                              train_size=10000, test_size=1000,
                                                              experiment=self.params['data_cat'],
                                                              pcap_file=self.anml_pcap)
                dump_data((self.flows, self.labels, self.subflow_interval), output_file=output_flows_labels,
                          verbose=True)
        except Exception as e:
            print(f'{e}')

        return self.flows, self.labels
