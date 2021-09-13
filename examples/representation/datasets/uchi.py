""" UCHI

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import glob
import os

import numpy as np
from loguru import logger as lg

from odet.datasets._base import Base
from odet.pparser.parser import _pcap2flows, _get_flow_duration, filter_ip, augment_flows, keep_mac_address
from odet.utils.tool import load, check_path, dump, get_file_path, remove_file


def get_iot2021_flows(in_dir=f'../Datasets/UCHI/IOT_2021/data-clean/refrigerator',
                      dataset_name='',
                      out_dir='',
                      direction='src'):
	""" Hard coding in_dir and pcap paths

	Note:
		1) refrigerator IP changes over time (dynamic ip), so here we filter with mac address.
		2) please don't merge all pcaps first and then obtain flows.
	Parameters
	----------
	in_dir
	direction

	Returns
	-------

	"""
	ip2device = {'192.168.143.152': 'refrigerator', }
	device2ip = {'refrigerator': '192.168.143.43', 'nestcam': '192.168.143.104', 'alexa': '192.168.143.74'}
	# #
	device2mac = {'refrigerator': '70:2c:1f:39:25:6e', 'nestcam': '18:b4:30:8a:9f:b2',
	              'alexa': '4c:ef:c0:0b:91:b3'}
	normal_pcaps = list(glob.iglob(in_dir + '/no_interaction/**/*.' + 'pcap', recursive=True))
	normal_pcaps.append(in_dir + '/idle_1629935923.pcap')
	normal_pcaps.append(in_dir + '/idle_1630275254.pcap')
	normal_pcaps = sorted(normal_pcaps)
	normal_flows = []
	for f in normal_pcaps:
		filter_f = f'{out_dir}/~tmp.pcap'
		check_path(filter_f)
		keep_mac_address(f, kept_ips=[device2mac['refrigerator']], out_file=filter_f, direction=direction)
		flows = _pcap2flows(filter_f, verbose=10)  # normal  flows
		normal_flows.extend(flows)
	lg.debug(f'total normal pcaps: {len(normal_pcaps)} and total flows: {len(normal_flows)}')

	# get abnormal flows
	abnormal_pcaps = list(glob.iglob(in_dir + '/open_close_fridge/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/put_back_item/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/screen_interaction/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/take_out_item/**/*.' + 'pcap', recursive=True))
	abnormal_pcaps = sorted(abnormal_pcaps)

	abnormal_flows = []
	for f in abnormal_pcaps:
		filter_f = f'{out_dir}/~tmp.pcap'
		check_path(filter_f)
		keep_mac_address(f, kept_ips=[device2mac['refrigerator']], out_file=filter_f, direction=direction)
		flows = _pcap2flows(filter_f, verbose=10)  # normal  flows
		abnormal_flows.extend(flows)
	lg.debug(f'total abnormal pcaps: {len(abnormal_pcaps)} and total flows: {len(abnormal_flows)}')

	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': normal_pcaps, 'abnormal_pcaps': abnormal_pcaps,
	        'device2mac': device2mac, 'filter_mac': device2mac['refrigerator'],
	        'direction': direction, 'in_dir': in_dir}
	return meta


def get_ghome2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019', out_dir='',
                        dataset_name='ghome_192.168.143.20',
                        direction='src'):
	IP = '192.168.143.20'
	normal_pcap = os.path.join(out_dir, f'pc_{IP}.pcap')
	check_path(normal_pcap)
	# file_name = 'google_home-2daysactiv-src_192.168.143.20-normal.pcap'
	file_name = 'fridge_cam_sound_ghome_2daysactiv-ghome_normal.pcap'
	pcap_file = get_file_path(in_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, normal_pcap, ips=[IP], direction=direction)
	normal_flows = _pcap2flows(normal_pcap, verbose=10)  # normal flows, ~4500
	max_interval = np.quantile([_get_flow_duration(pkts) for f, pkts in normal_flows], q=0.9)
	normal_flows = augment_flows(normal_flows, step=10, max_interval=max_interval)

	abnormal_pcap = os.path.join(out_dir, f'pc_{IP}_abnormal.pcap')
	check_path(normal_pcap)
	# file_name = 'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap'
	file_name = 'fridge_cam_sound_ghome_2daysactiv-ghome_abnormal.pcap'
	pcap_file = get_file_path(ipt_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, abnormal_pcap, ips=[IP], direction=direction)
	abnormal_flows = _pcap2flows(abnormal_pcap, verbose=10)  # abnormal  flows
	abnormal_flows = augment_flows(abnormal_flows, step=10, max_interval=max_interval)
	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': [normal_pcap], 'abnormal_pcaps': [abnormal_pcap],
	        'direction': direction, 'in_dir': in_dir}
	return meta


def get_scam2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019', out_dir='',
                       dataset_name='scam_192.168.143.42',
                       direction='src'):
	IP = '192.168.143.42'
	normal_pcap = os.path.join(out_dir, f'pc_{IP}.pcap')
	check_path(normal_pcap)
	file_name = 'fridge_cam_sound_ghome_2daysactiv-scam_normal.pcap'
	pcap_file = get_file_path(in_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, normal_pcap, ips=[IP], direction=direction)
	normal_flows = _pcap2flows(normal_pcap, verbose=10)  # ~1000 normal flows, it will generate > 1000 subflows
	max_interval = np.quantile([_get_flow_duration(pkts) for f, pkts in normal_flows], q=0.9)
	normal_flows = augment_flows(normal_flows, step=10, max_interval=max_interval)
	lg.debug(f'normal_flows: {len(normal_flows)}')

	abnormal_pcap = os.path.join(out_dir, f'pc_{IP}_abnormal.pcap')
	check_path(normal_pcap)
	# file_name = 'samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pca'
	file_name = 'fridge_cam_sound_ghome_2daysactiv-scam_abnormal.pcap'
	pcap_file = get_file_path(ipt_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, abnormal_pcap, ips=[IP], direction=direction)
	abnormal_flows = _pcap2flows(abnormal_pcap, verbose=10)
	abnormal_flows = augment_flows(abnormal_flows, step=1, max_interval=max_interval)
	lg.debug(f'after augmenting abnormal_flows: {len(abnormal_flows)}')
	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': [normal_pcap], 'abnormal_pcaps': [abnormal_pcap],
	        'direction': direction, 'in_dir': in_dir}
	return meta


def get_bstch2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019', out_dir='',
                        dataset_name='scam_192.168.143.48',
                        direction='src'):
	IP = '192.168.143.48'
	normal_pcap = os.path.join(out_dir, f'pc_{IP}.pcap')
	check_path(normal_pcap)
	# file_name = 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap'
	file_name = 'fridge_cam_sound_ghome_2daysactiv-bstch_normal.pcap'
	pcap_file = get_file_path(in_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, normal_pcap, ips=[IP], direction=direction)
	normal_flows = _pcap2flows(normal_pcap, verbose=10)  # normal  flows
	max_interval = np.quantile([_get_flow_duration(pkts) for f, pkts in normal_flows], q=0.9)
	normal_flows = augment_flows(normal_flows, step=10, max_interval=max_interval)

	abnormal_pcap = os.path.join(out_dir, f'pc_{IP}_abnormal.pcap')
	check_path(normal_pcap)
	# file_name = 'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap'
	file_name = 'fridge_cam_sound_ghome_2daysactiv-bstch_abnormal.pcap'
	pcap_file = get_file_path(ipt_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, abnormal_pcap, ips=[IP], direction=direction)
	abnormal_flows = _pcap2flows(abnormal_pcap, verbose=10)  # abnormal  flows
	# abnormal_flows = augment_flows(abnormal_flows, starts=50, max_len=max_len)
	abnormal_flows = augment_flows(abnormal_flows, step=10, max_interval=max_interval)
	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': [normal_pcap], 'abnormal_pcaps': [abnormal_pcap],
	        'direction': direction, 'in_dir': in_dir}
	return meta


def get_smtv2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019', out_dir='',
                       dataset_name='smtv_10.42.0.1',
                       direction='src'):
	IP = '10.42.0.1'
	normal_pcap = os.path.join(out_dir, f'pc_{IP}.pcap')
	check_path(normal_pcap)
	file_name = 'pc_10.42.0.1_normal.pcap'
	pcap_file = get_file_path(in_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, normal_pcap, ips=[IP], direction=direction)
	normal_flows = _pcap2flows(normal_pcap, verbose=10)  # normal  flows
	max_interval = np.quantile([_get_flow_duration(pkts) for f, pkts in normal_flows], q=0.9)
	normal_flows = augment_flows(normal_flows, step=10, max_interval=max_interval)

	abnormal_pcap = os.path.join(out_dir, f'pc_10.42.0.119_abnormal.pcap')
	check_path(normal_pcap)
	file_name = 'pc_10.42.0.119_anomaly.pcap'
	pcap_file = get_file_path(in_dir=in_dir, dataset_name=dataset_name,
	                          file_name=file_name)
	filter_ip(pcap_file, abnormal_pcap, ips=['10.42.0.119'], direction=direction)
	abnormal_flows = _pcap2flows(abnormal_pcap, verbose=10)  # normal  flows
	abnormal_flows = augment_flows(abnormal_flows, step=10, max_interval=max_interval)
	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': [normal_pcap], 'abnormal_pcaps': [abnormal_pcap],
	        'direction': direction, 'in_dir': in_dir}
	return meta


class UCHI(Base):
	def __init__(self, in_dir='../Datasets', dataset_name='UCHI(SFRIG)', direction='src',
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
		lg.info(f'{self.Xy_file}: {os.path.exists(self.Xy_file)}')

	def generate(self):
		remove_file(self.Xy_file, self.overwrite)

		if os.path.exists(self.Xy_file):
			Xy_meta = load(self.Xy_file)
		else:
			if self.dataset_name in ['UCHI(SFRIG_2021)', 'UCHI(SMTV_2019)', 'UCHI(GHOME_2019)',
			                         'UCHI(SCAM_2019)', 'UCHI(BSTCH_2019)']:
				self._generate_pcap()  # generate data
				flows_meta = self._generate_flows()  # normal_abnormal.data
				# Xy (fixed feature data)
				Xy_meta = self._generate_features(flows_meta['normal_flows'], flows_meta['abnormal_flows'])
			else:
				msg = f'{self.dataset_name}'
				raise NotImplementedError(msg)
		self.X, self.y = Xy_meta['X'], Xy_meta['y']
		self.Xy_meta = Xy_meta
		return self.Xy_meta

	def _generate_pcap(self):
		regenerate = False
		# step 1: obtain pcap and label
		if self.dataset_name == 'UCHI(SFRIG_2021)':
			self.IP = 'mac_70:2c:1f:39:25:6e'  # IP for the new data changes over time, so here use mac address instead
			self.orig_flows = os.path.join(self.out_dir, f'iot2021-orig_sfrig_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.Xy_file, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				# hard coding (is not a good idea)
				meta = get_iot2021_flows(in_dir=f'../Datasets/UCHI/IOT_2021/data-clean/refrigerator',
				                         dataset_name=self.dataset_name,
				                         out_dir=self.out_dir,
				                         direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				regenerate = True
			else:
				pass
		elif self.dataset_name == 'UCHI/IOT_2019/ghome_192.168.143.20' or self.dataset_name == 'UCHI(GHOME_2019)':
			self.IP = '192.168.143.20'
			self.orig_flows = os.path.join(self.out_dir, f'ghome2019-orig_sfrig_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.Xy_file, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = get_ghome2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019/',
				                           dataset_name='ghome_192.168.143.20',
				                           out_dir=self.out_dir,
				                           direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				regenerate = True
			else:
				pass
		elif self.dataset_name == 'UCHI/IOT_2019/scam_192.168.143.42' or self.dataset_name == 'UCHI(SCAM_2019)':
			self.IP = '192.168.143.42'
			self.orig_flows = os.path.join(self.out_dir, f'scam2019-orig_scam_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.Xy_file, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = get_scam2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019/',
				                          dataset_name='scam_192.168.143.42',
				                          out_dir=self.out_dir,
				                          direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				regenerate = True
			else:
				pass
		elif self.dataset_name == 'UCHI/IOT_2019/bstch_192.168.143.48' or self.dataset_name == 'UCHI(BSTCH_2019)':
			self.IP = '192.168.143.48'
			self.orig_flows = os.path.join(self.out_dir, f'bstch2019-orig_bstch_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.Xy_file, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = get_bstch2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019/',
				                           dataset_name='bstch_192.168.143.48',
				                           out_dir=self.out_dir,
				                           direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				regenerate = True
			else:
				pass
		elif self.dataset_name == 'UCHI/IOT_2019/smtv_10.42.0.1' or self.dataset_name == 'UCHI(SMTV_2019)':
			self.IP = '10.42.0.1'
			self.orig_flows = os.path.join(self.out_dir, f'smtv2019-orig_smtv_{self.direction}_flows-{self.IP}.dat')
			remove_file(self.Xy_file, self.overwrite)
			if not os.path.exists(self.orig_flows):
				lg.warning(f'{self.orig_flows} does not exist.')
				check_path(self.orig_flows)
				meta = get_smtv2019_flows(in_dir=f'../Datasets/UCHI/IOT_2019/',
				                          dataset_name='smtv_10.42.0.1',
				                          out_dir=self.out_dir,
				                          direction=self.direction)
				dump(meta, out_file=self.orig_flows)
				regenerate = True
			else:
				pass
		else:
			raise ValueError('dataset does not exist.')

# @timer
# def _generate_flows(self):
#
# 	self.subflows_file = os.path.join(self.out_dir, 'normal_abnormal_subflows.dat')
# 	remove_file(self.subflows_file, self.overwrite)
#
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
