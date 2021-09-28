""" Base class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os

from loguru import logger as lg

from odet.datasets._base import Base
from odet.pparser.parser import filter_ip, _pcap2flows
from odet.utils.tool import load, get_file_path, check_path, dump, remove_file


class MAWI(Base):

	def __init__(self, in_dir='../Datasets', dataset_name='MAWI', direction='src',
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
		lg.info(f'{self.Xy_file} exists: {os.path.exists(self.Xy_file)}')

	def generate(self):
		remove_file(self.Xy_file, self.overwrite)

		if os.path.exists(self.Xy_file):
			Xy_meta = load(self.Xy_file)
		else:
			if self.dataset_name in ['MAWI']:
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
		if self.dataset_name == 'MAWI/WIDE_2019/pc_202.171.168.50' or self.dataset_name == 'MAWI':
			# "http://mawi.wide.ad.jp/mawi/samplepoint-F/2019/201912071400.html"
			self.IP = '202.171.168.50'
			self.orig_flows = os.path.join(self.out_dir, f'mawi_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.orig_flows, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = self.get_mawi_flows(in_dir=f'../Datasets', direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				lg.debug(f'in_dir (pcaps): ' + meta['in_dir'] + ', direction: ' + meta['direction'])
				lg.debug(f'normal_pcap: ' + str(len(meta['normal_pcap'])) + ', normal_flows: '
				         + str(len(meta['normal_flows'])))
				lg.debug(f'abnormal_pcap: ' + str(len(meta['abnormal_pcap'])) + ', abnormal_flows: '
				         + str(len(meta['abnormal_flows'])))

		else:
			raise ValueError('dataset does not exist.')

	def get_mawi_flows(self, in_dir='../Datatsets', direction='src'):

		self.normal_pcap = os.path.join(self.out_dir, f'pc_202.171.168.50.pcap')
		check_path(self.normal_pcap)
		file_name = 'samplepoint-F_201912071400-src_dst_202.171.168.50.pcap'
		pcap_file = get_file_path(in_dir=in_dir, dataset_name='MAWI/WIDE_2019',
		                          file_name=file_name)
		filter_ip(pcap_file, self.normal_pcap, ips=['202.171.168.50'], direction=direction)
		normal_flows = _pcap2flows(self.normal_pcap, verbose=10)  # normal  flows

		self.abnormal_pcap = os.path.join(self.out_dir, f'pc_203.113.113.16_abnormal.pcap')
		check_path(self.normal_pcap)
		# file_name = 'samplepoint-F_201912071400-src_dst_202.4.27.109.pcap'    # ~5000
		file_name = 'samplepoint-F_201912071400-src_203.113.113.16.pcap'  # ~1500
		pcap_file = get_file_path(ipt_dir=in_dir, dataset_name='MAWI/WIDE_2019',
		                          file_name=file_name)
		filter_ip(pcap_file, self.abnormal_pcap, ips=['203.113.113.16'], direction=direction)
		abnormal_flows = _pcap2flows(self.abnormal_pcap, verbose=10)  # normal  flows
		meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
		        'normal_pcap': self.normal_pcap, 'abnormal_pcap': self.abnormal_pcap,
		        'direction': direction, 'in_dir': in_dir}
		return meta

# @timer
# def _generate_flows(self):
# 	self.subflows_file = os.path.join(self.out_dir, 'normal_abnormal_subflows.dat')
# 	remove_file(self.subflows_file, self.overwrite)
# 	if os.path.exists(self.subflows_file):
# 		return load(self.subflows_file)
#
# 	# step 2: extract flows from pcap
# 	##############################################################################################
# 	meta = load(self.orig_flows)
# 	normal_flows, abnormal_flows = meta['normal_flows'], meta['abnormal_flows']
# 	lg.debug(f'original normal flows: {len(normal_flows)} and abnormal flows: {len(abnormal_flows)}')
# 	qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
# 	len_stat = np.quantile([len(pkts) for f, pkts in normal_flows], q=qs)
# 	lg.debug(f'flows: {len(normal_flows)}, length statistic: {len_stat}, when q = {qs}')
# 	meta = {'flows': normal_flows, 'len_stat': (len_stat, qs),
# 	        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows}
# 	dump(meta, out_file=os.path.join(self.out_dir, 'normal_abnormal_flows.dat'))
#
# 	# step 2.2. only get normal flows durations
# 	self.flows_durations = [_get_flow_duration(pkts) for (fids, pkts) in normal_flows]
# 	normal_durations_stat = np.quantile(self.flows_durations, q=qs)
# 	lg.debug(f'normal_durations_stat: {normal_durations_stat}')
# 	self.subflow_interval = np.quantile(self.flows_durations, q=self.q_flow_dur)  # median  of flow_durations
# 	lg.debug(f'---subflow_interval: {self.subflow_interval}, q_flow_dur: {self.q_flow_dur}')
# 	# step 2.3 get subflows
# 	normal_flows, _ = _flows2subflows(normal_flows, interval=self.subflow_interval,
# 	                                  labels=['0'] * len(normal_flows))
# 	abnormal_flows, _ = _flows2subflows(abnormal_flows, interval=self.subflow_interval,
# 	                                    labels=['1'] * len(abnormal_flows))
# 	lg.debug(f'normal_flows: {len(normal_flows)}, and abnormal_flows: {len(abnormal_flows)} '
# 	         f'with interval: {self.subflow_interval} and q: {self.q_flow_dur}')
# 	meta = {'normal_flows_durations': self.flows_durations, 'normal_durations_stat': (normal_durations_stat, qs),
# 	        'subflow_interval': self.subflow_interval, 'q_flow_dur': self.q_flow_dur,
# 	        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows}
# 	dump(meta, out_file=self.subflows_file)
#
# 	# only return subflows
# 	return meta
