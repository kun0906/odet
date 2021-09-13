""" Base class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import os

import numpy as np
from loguru import logger as lg

from odet.datasets._base import Base
from odet.pparser.parser import filter_ip, _pcap2flows, _flows2subflows, \
	_get_flow_duration
from odet.utils.tool import load, get_file_path, check_path, dump, timer, remove_file


class CTU(Base):

	def __init__(self, in_dir='../Datasets', dataset_name='CTU', direction='src',
	             out_dir='.', feature_name='IAT+SIZE', header=False, q_flow_dur=0.9,
	             overwrite=False, random_state=42):
		self.X = None
		self.y = None
		self.in_dir = in_dir
		self.overwrite = overwrite
		self.out_dir = out_dir
		self.feature_name = feature_name
		self.header = header
		self.dataset_name = dataset_name
		self.direction = direction
		self.random_state = random_state
		self.q_flow_dur = q_flow_dur

		self.Xy_file = os.path.join(self.out_dir, self.direction, self.dataset_name,
		                            self.feature_name, f'header_{self.header}', f'Xy.dat')
		self.out_dir = os.path.join(self.out_dir, self.direction, self.dataset_name)
		lg.info(f'{self.Xy_file}')

	def generate(self):
		remove_file(self.Xy_file, self.overwrite)
		if os.path.exists(self.Xy_file):
			Xy_meta = load(self.Xy_file)
		else:
			if self.dataset_name in ['CTU']:
				self._generate_pcap()  # generate data
				flows_meta = self._generate_flows()  # normal_abnormal.data
				# Xy (fixed feature data)
				Xy_meta = self._generate_features(flows_meta['normal_flows'], flows_meta['abnormal_flows'])
			else:
				msg = f'{self.dataset_name}'
				raise NotImplementedError(msg)
		self.X, self.y = Xy_meta['X'], Xy_meta['y']

		return Xy_meta

	def _generate_pcap(self):
		# preprocessed the pcap and label on original pcap and label
		if self.dataset_name == 'CTU/IOT_2017/pc_192.168.1.196' or self.dataset_name == 'CTU':
			self.IP = '192.168.1.196'
			self.orig_flows = os.path.join(self.out_dir, f'ctu_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.orig_flows, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = self.get_ctu_flows(in_dir=f'../Datasets', direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				lg.debug(f'in_dir (pcaps): ' + meta['in_dir'] + ', direction: ' + meta['direction'])
				lg.debug(f'normal_pcap: ' + str(len(meta['normal_pcap'])) + ', normal_flows: '
				         + str(len(meta['normal_flows'])))
				lg.debug(f'abnormal_pcap: ' + str(len(meta['abnormal_pcap'])) + ', abnormal_flows: '
				         + str(len(meta['abnormal_flows'])))

		else:
			raise ValueError('dataset does not exist.')

	def get_ctu_flows(self, in_dir='../Datatsets', direction='src'):
		"""
		https://www.stratosphereips.org/datasets-iot
		Malware on IoT Dataset
		"""
		self.normal_pcap = os.path.join(self.out_dir, f'pc_192.168.1.196.pcap')
		check_path(self.normal_pcap)
		# filter pcap
		# file_name = '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
		file_name = 'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap'
		pcap_file = get_file_path(in_dir=in_dir, dataset_name='CTU/IOT_2017',
		                          file_name=file_name)
		filter_ip(pcap_file, self.normal_pcap, ips=['192.168.1.196'], direction=direction)
		normal_flows = _pcap2flows(self.normal_pcap, verbose=10)  # normal  flows

		self.abnormal_pcap = os.path.join(self.out_dir, f'pc_192.168.1.195_abnormal.pcap')
		check_path(self.normal_pcap)
		# file_name = '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap'
		file_name = 'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap'
		pcap_file = get_file_path(ipt_dir=in_dir, dataset_name='CTU/IOT_2017',
		                          file_name=file_name)
		filter_ip(pcap_file, self.abnormal_pcap, ips=['192.168.1.195'], direction=direction)
		abnormal_flows = _pcap2flows(self.abnormal_pcap, verbose=10)  # normal  flows
		meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
		        'normal_pcap': self.normal_pcap, 'abnormal_pcap': self.abnormal_pcap,
		        'direction': direction, 'in_dir': in_dir}
		return meta

