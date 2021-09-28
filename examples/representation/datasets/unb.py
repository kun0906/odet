""" UNB

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import os

import numpy as np
import pandas as pd
from loguru import logger as lg

from odet.datasets._base import Base
from odet.pparser.parser import filter_ip, filter_csv_ip, _pcap2flows, _get_flow_duration, augment_flows
from odet.utils.tool import load, get_file_path, check_path, dump, remove_file


def merge_labels(label_file_lst=[], mrg_label_path=''):
	"""merge csvs

	Parameters
	----------
	label_file_lst
	mrg_label_path

	Returns
	-------

	"""
	with open(mrg_label_path, 'w') as out_f:
		header = True
		for i, label_file in enumerate(label_file_lst):
			with open(label_file, 'r') as in_f:
				line = in_f.readline()
				while line:
					if line.strip().startswith('Flow ID') and header:
						if header:
							header = False
							print(line)
							out_f.write(line.strip('\n') + '\n')
						else:
							pass
						line = in_f.readline()
						continue
					if line.strip() == '':
						line = in_f.readline()
						continue
					out_f.write(line.strip('\n') + '\n')
					line = in_f.readline()
	return mrg_label_path


def split_normal_abnormal(flows, labels):
	""" split normal and abnormal flows

	Parameters
	----------
	flows: (fid, pkts)
	labels: (fid, label)

	Returns
	-------
	normal_flows: list
	abnormal_flows: list

	"""

	label_mp = {}
	conflict = set()
	for vs in labels:
		# fid = tuple(sorted([v if i < 2 else int(v) for i, v in enumerate(fid.split('-'))]))
		# fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
		fid = tuple([str(vs[1]), str(vs[3]), str(vs[2]), str(vs[4]), str(vs[5])])
		label = 'normal' if str(vs[-1]).lower() in ['normal', 'benign'] else 'abnormal'
		# lg.debug(fid, label)
		if fid in label_mp.keys():
			if label_mp[fid] != label:
				conflict.add((fid, label, label_mp[fid]))
		else:
			label_mp[fid] = label
	lg.debug(f'tot labels: {len(labels)}, where label_mp: {len(label_mp)} and conflict: {len(conflict)} ({conflict})')

	normal_flows = []
	abnormal_flows = []
	others = []
	for fid, pkts in flows:
		# fid = tuple(sorted([str(v) for v in fid]))    # might be wrong
		fid = tuple([str(v) for v in fid])
		if fid in label_mp.keys():
			label = label_mp[fid]
			if label.lower() in ['normal', 'benign']:
				normal_flows.append((fid, pkts))
			else:
				abnormal_flows.append((fid, pkts))
		else:  # fid doesn't exist in label_mp
			others.append((fid, pkts))
	lg.debug(f'tot flows: {len(flows)}, where normal: {len(normal_flows)}, abnormal: {len(abnormal_flows)} '
	         f'and others: {len(others)}')
	return normal_flows, abnormal_flows


class UNB(Base):

	def __init__(self, in_dir='../Datasets', dataset_name='UNB(PC1)', direction='src',
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
			if self.dataset_name in ['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)']:
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

		# step 1: obtain pcap and label
		if self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.5' or self.dataset_name == 'UNB(PC1)':
			self.IP = '192.168.10.5'
			self.orig_flows = os.path.join(self.out_dir, f'orig_unb(pc1)_{self.direction}_flows-{self.IP}.dat')
		elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.8' or self.dataset_name == 'UNB(PC2)':
			self.IP = '192.168.10.8'
			self.orig_flows = os.path.join(self.out_dir, f'orig_unb(pc2)_{self.direction}_flows-{self.IP}.dat')
		elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.9' or self.dataset_name == 'UNB(PC3)':
			self.IP = '192.168.10.9'
			self.orig_flows = os.path.join(self.out_dir, f'orig_unb(pc3)_{self.direction}_flows-{self.IP}.dat')
		elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.14' or self.dataset_name == 'UNB(PC4)':
			self.IP = '192.168.10.14'
			self.orig_flows = os.path.join(self.out_dir, f'orig_unb(pc4)_{self.direction}_flows-{self.IP}.dat')
		elif self.dataset_name == 'UNB/CICIDS_2017/pc_192.168.10.15' or self.dataset_name == 'UNB(PC5)':
			self.IP = '192.168.10.15'
			self.orig_flows = os.path.join(self.out_dir, f'orig_unb(pc5)_{self.direction}_flows-{self.IP}.dat')
		elif self.dataset_name == 'DEMO_IDS/DS-srcIP_192.168.10.5':
			self.IP = '192.168.10.5'
			self.orig_flows = os.path.join(self.out_dir, f'orig_demo_{self.direction}_flows-{self.IP}.dat')
		else:
			raise ValueError('dataset does not exist.')

		remove_file(self.Xy_file, self.overwrite)
		if not os.path.exists(self.orig_flows):
			lg.warning(f'{self.orig_flows} does not exist.')
			check_path(self.orig_flows)
			meta = self.get_unb_flows(in_dir=f'../Datasets', direction=self.direction)
			dump(meta, out_file=self.orig_flows)
			lg.debug(f'in_dir (pcaps): ' + meta['in_dir'] + ', direction: ' + meta['direction'])
			lg.debug(f'normal_pcap: ' + str(len(meta['normal_pcap'])) + ', normal_flows: '
			         + str(len(meta['normal_flows'])))
			lg.debug(f'abnormal_pcap: ' + str(len(meta['abnormal_pcap'])) + ', abnormal_flows: '
			         + str(len(meta['abnormal_flows'])))
		else:
			pass

	def get_unb_flows(self, in_dir='../Datatsets', direction='src'):

		# preprocessed the pcap and label on original pcap and label
		self.pcap_file = os.path.join(self.out_dir, f'pc_{self.IP}_AGMT.pcap')
		self.label_file = os.path.join(self.out_dir, f'pc_{self.IP}_AGMT.csv')
		remove_file(self.pcap_file, self.overwrite)
		remove_file(self.label_file, self.overwrite)
		check_path(self.pcap_file)
		check_path(self.label_file)

		if not os.path.exists(self.pcap_file) or not os.path.exists(self.label_file):
			# 1. original pcap
			friday_pacp_orig = get_file_path(ipt_dir=in_dir,
			                                 dataset_name='UNB/CICIDS_2017/',
			                                 data_cat='pcaps/Friday',
			                                 file_name='Friday-WorkingHours.pcap')
			# filter pcap
			filter_ip(friday_pacp_orig, out_file=self.pcap_file, ips=[self.IP],
			          direction=self.direction,
			          keep_original=True)

			# 2. merge original labels
			friday_label = get_file_path(ipt_dir=self.out_dir,
			                             dataset_name='UNB/CICIDS_2017/',
			                             data_cat='labels/Friday',
			                             file_name='Friday-WorkingHours-Morning.pcap_ISCX.csv')
			friday_label_orig1 = get_file_path(ipt_dir=in_dir,
			                                   dataset_name='UNB/CICIDS_2017/',
			                                   data_cat='labels/Friday',
			                                   file_name='Friday-WorkingHours-Morning.pcap_ISCX.csv')
			friday_label_orig2 = get_file_path(ipt_dir=in_dir,
			                                   dataset_name='UNB/CICIDS_2017/',
			                                   data_cat='labels/Friday',
			                                   file_name='Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
			friday_label_orig3 = get_file_path(ipt_dir=in_dir,
			                                   dataset_name='UNB/CICIDS_2017/',
			                                   data_cat='labels/Friday',
			                                   file_name='Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
			friday_label_tmp = friday_label + '-all.csv'
			check_path(friday_label_tmp)
			merge_labels([friday_label_orig1, friday_label_orig2, friday_label_orig3],
			             mrg_label_path=friday_label_tmp)
			filter_csv_ip(friday_label_tmp, out_file=self.label_file, ips=[self.IP], direction=self.direction)

		##############################################################################################
		# step 2.1 extract flows
		flows = _pcap2flows(self.pcap_file, verbose=10)  # normal and abnormal flows
		# step 2.2 split normal flow and abnormal flow
		labels = pd.read_csv(self.label_file).values  #
		normal_flows, abnormal_flows = split_normal_abnormal(flows, labels)
		# augment abnormal flows
		max_interval = np.quantile([_get_flow_duration(pkts) for f, pkts in normal_flows], q=0.9)
		abnormal_flows = augment_flows(abnormal_flows, step=1, max_interval=max_interval)
		meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
		        'normal_pcap': self.pcap_file, 'abnormal_pcap': self.label_file,
		        'direction': direction, 'in_dir': in_dir}

		return meta

# @timer
# def _generate_flows(self):
#
# 	self.subflows_file = os.path.join(self.out_dir, 'normal_abnormal_subflows.dat')
# 	remove_file(self.subflows_file, self.overwrite)
# 	if os.path.exists(self.subflows_file):
# 		return load(self.subflows_file)
#
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
# 	meta = {'normal_flows_durations': self.flows_durations, 'normal_durations_stat': (normal_durations_stat, qs),
# 	        'subflow_interval': self.subflow_interval, 'q_flow_dur': self.q_flow_dur,
# 	        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows}
# 	dump(meta, out_file=self.subflows_file)
#
# 	# only return subflows
# 	return meta
