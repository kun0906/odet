"""Get feature data


	Get basic info of pcap
	Split pcap:
	    editcap -A “2017-07-07 9:00” -B ”2017-07-07 12:00” Friday-WorkingHours.pcap Friday-WorkingHours_09_00-12_00.pcap
	    editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap
	    # only save the first 10000 packets
	    editcap -r AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap 0-10000

	filter:
	   cmd = f"tshark -r {in_file} -w {out_file} {srcIP_str}"
"""

import os
import subprocess
import numpy as np

import os.path as pth
from collections import Counter
from shutil import copyfile

import sklearn
from loguru import logger as lg
from odet.pparser.parser import PCAP, _get_flow_duration, _get_split_interval, _flows2subflows, _get_IAT, _get_SIZE, \
	_get_IAT_SIZE, _get_STATS, _get_SAMP_NUM, _get_SAMP_SIZE, _get_FFT_data, _get_header_features
from sklearn.utils import shuffle

from examples.representation._constants import *
# from odet.datasets.uchi import split_by_activity
from odet.utils.tool import data_info, load, dump, timer, time_func


def keep_ip(in_file, out_file='', kept_ips=[''], direction='src_dst'):
	if out_file == '':
		ips_str = '-'.join(kept_ips)
		out_file = os.path.splitext(in_file)[0] + f'-src_{ips_str}.pcap'  # Split a path in root and extension.
	if os.path.exists(out_file):
		return out_file
	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))
	lg.debug(out_file)
	# only keep srcIPs' traffic
	if direction == 'src':
		srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
	else:  # default
		srcIP_str = " or ".join([f'ip.addr=={srcIP}' for srcIP in kept_ips])
	cmd = f"tshark -r {in_file} -w {out_file} {srcIP_str}"

	lg.debug(f'{cmd}')
	try:
		result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
	except Exception as e:
		lg.debug(f'{e}, {result}')
		return -1

	return out_file


def keep_csv_ip(label_file, out_file, ips=[], direction='src_dst', header=True, keep_original=True, verbose=10):
	# from shutil import copyfile
	# copyfile(label_file, out_file)

	# lg.debug(label_file_lst, mrg_label_path)
	# if os.path.exists(mrg_label_path):
	#     os.remove(mrg_label_path)

	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))

	with open(out_file, 'w') as out_f:
		with open(label_file, 'r') as in_f:
			line = in_f.readline()
			while line:
				if line.strip().startswith('Flow') and header:
					if header:
						header = False
						lg.debug(line)
						out_f.write(line.strip('\n') + '\n')
					else:
						pass
					line = in_f.readline()
					continue
				if line.strip() == '':
					line = in_f.readline()
					continue

				exist = False
				for ip in ips:
					if ip in line:
						exist = True
						break
				if exist:
					out_f.write(line.strip('\n') + '\n')
				line = in_f.readline()

	return out_file


def merge_pcaps(in_files, out_file):
	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))
	cmd = f"mergecap -w \"{out_file}\" " + ' '.join(in_files)

	lg.debug(f'{cmd}')
	try:
		result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
	except Exception as e:
		lg.debug(f'{e}, {result}')
		return -1

	return out_file


def merge_csvs(in_files=[], out_file=''):
	lg.debug(in_files, out_file)
	if os.path.exists(out_file):
		os.remove(out_file)

	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))

	with open(out_file, 'w') as out_f:
		header = True
		for i, label_file in enumerate(in_files):
			with open(label_file, 'r') as in_f:
				line = in_f.readline()
				while line:
					if line.strip().startswith('Flow ID') and header:
						if header:
							header = False
							lg.debug(line)
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

	return out_file


def flows2subflows_SCAM(full_flows, interval=10, num_pkt_thresh=2, data_name='', abnormal=False):
	from scapy.layers.inet import UDP, TCP, IP
	remainder_cnt = 0
	new_cnt = 0  # a flow is not split by an intervals
	flows = []  # store the subflows
	step_flows = []
	tmp_arr2 = []
	tmp_arr1 = []
	lg.debug(f'interval: {interval}')
	lg.debug('normal file does not need split with different steps, only anomaly file needs.')
	for i, (fid, pkts) in enumerate(full_flows):
		times = [float(pkt.time) for pkt in pkts]
		if i % 1000 == 0:
			lg.debug(f'session_i: {i}, len(pkts): {len(pkts)}')

		flow_type = None
		new_flow = 0
		dur = max(times) - min(times)
		if dur >= 2 * interval:
			tmp_arr2.append(max(times) - min(times))  # 10% flows exceeds the interals

		if dur >= 1 * interval:
			tmp_arr1.append(max(times) - min(times))
		step = 0  # 'step' for 'normal data' always equals 0. If datasets needs to be agumented, then slide window with step
		while step < len(pkts):
			# lg.debug(f'i: {i}, step:{step}, len(pkts[{step}:]): {len(pkts[step:])}')
			dur_tmp = max(times[step:]) - min(times[step:])
			if dur_tmp <= interval:
				if step == 0:
					subflow = [(float(pkt.time), pkt) for pkt in pkts[step:]]
					step_flows.append((fid, subflow))
					flows.append((fid, subflow))
				break  # break while loop
			flow_i = []
			subflow = []
			for j, pkt in enumerate(pkts[step:]):
				if TCP not in pkt and UDP not in pkt:
					break
				if j == 0:
					flow_start_time = float(pkt.time)
					subflow = [(float(pkt.time), pkt)]
					split_flow = False  # if a flow is not split with interval, label it as False, otherwise, True
					continue
				# handle TCP packets
				if IP in pkt and TCP in pkt:
					flow_type = 'TCP'
					fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
					if float(pkt.time) - flow_start_time > interval:
						flow_i.append((fid, subflow))
						flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
						subflow = [(float(pkt.time), pkt)]
						split_flow = True
					else:
						subflow.append((float(pkt.time), pkt))

				# handle UDP packets
				elif IP in pkt and UDP in pkt:
					# parse 5-tuple flow ID
					fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
					flow_type = 'UDP'
					if float(pkt.time) - flow_start_time > interval:
						flow_i.append((fid, subflow))
						flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
						subflow = [(float(pkt.time), pkt)]
						split_flow = True
					else:
						subflow.append((float(pkt.time), pkt))

			if (split_flow == False) and (flow_type in ['TCP', 'UDP']):
				new_cnt += 1
				flow_i.append((fid, subflow))
			else:
				# drop the last subflow after splitting a flow
				remainder_cnt += 1
			# flow_i.append((fid, subflow)) # don't include the remainder
			# lg.debug(i, new_flow, subflow)

			# drop the last one which interval is less than interval
			if step == 0:
				flows.extend(flow_i)

			step_flows.extend(flow_i)
			if data_name.upper() in ['DS60_UChi_IoT', 'SCAM1', 'scam1', 'GHOM1',
			                         'SFRIG1'] and abnormal:  # only augment abnormal flows
				step += 5  # 10 is the step for sampling, 'agument' anomaly files in DS60
			else:
				break

	lg.debug(
		f'tmp_arr2: {len(tmp_arr2)},tmp_arr1: {len(tmp_arr1)}, all_flows: {len(full_flows)}, subflows: {len(flows)}, step_flows: {len(step_flows)}, {data_name}, remain_subflow: {len(subflow)}')

	# sort all flows by packet arrival time, each flow must have at least two packets
	flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in flows if
	         len(times_pkts) >= max(2, num_pkt_thresh)]
	flows = [(fid, pkts) for fid, times, pkts in flows]
	lg.debug(f'the final subflows: len(flows): {len(flows)}, each of them has more than 2 pkts.')

	# sort all flows by packet arrival time, each flow must have at least two packets
	step_flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in step_flows if
	              len(times_pkts) >= max(2, num_pkt_thresh)]
	step_flows = [(fid, pkts) for fid, times, pkts in step_flows]
	lg.debug(f'the final step_flows: len(step_flows): {len(step_flows)}, each of them has more than 2 pkts.')
	if abnormal:
		return step_flows

	return flows


def _pcap2fullflows(pcap_file='', label_file=None, label='normal'):
	pp = PCAP(pcap_file=pcap_file)
	pp.pcap2flows()

	if label_file is None:
		pp.labels = [label] * len(pp.flows)
	else:
		pp.label_flows(label_file, label=label)
	flows = pp.flows
	labels = pp.labels
	lg.debug(f'labels: {Counter(labels)}')

	if label_file is not None:
		normal_flows = []
		normal_labels = []
		abnormal_flows = []
		abnormal_labels = []
		for (f, l) in zip(flows, labels):
			# lg.debug(l, 'normal' in l, 'abnormal' in l)
			if l.startswith('normal'):  # 'normal' in l and 'abnormal' in l both will return True
				normal_flows.append(f)
				normal_labels.append(l)
			else:
				# lg.debug(l)
				abnormal_flows.append(f)
				abnormal_labels.append(l)
	else:
		if label == 'normal':  # 'normal' in label:
			normal_flows = flows
			normal_labels = labels
			abnormal_flows = None
			abnormal_labels = None
		else:
			normal_flows = None
			normal_labels = None
			abnormal_flows = flows
			abnormal_labels = labels

	return normal_flows, normal_labels, abnormal_flows, abnormal_labels

class New_PCAP(PCAP):
	def __init__(self, pcap_file='xxx.pcap', *, flow_pkts_thres=2, verbose=10, random_state=42, sampling_rate=0.1):
		super(New_PCAP, self).__init__()
		# self.q_samps = q_samp
		self.verbose = verbose
		self.random_state = random_state
		self.flow_pkts_thres = flow_pkts_thres
		self.sampling_rate = sampling_rate

	@timer
	def _flow2features(self, feat_type='IAT', *, fft=False, header=False, dim=None):
		"""Extract features from each flow according to feat_type, fft and header.

		Parameters
		----------
		feat_type: str (default is 'IAT')
			which features do we want to extract from flows

		fft: boolean (default is False)
			if we need fft-features

		header: boolean (default is False)
			if we need header+features
		dim: dim of the "SIZE" feature

		Returns
		-------
			self
		"""
		self.feat_type = feat_type

		if dim is None:
			num_pkts = [len(pkts) for fid, pkts in self.flows]
			dim = int(np.floor(np.quantile(num_pkts, self.q_interval)))  # use the same q_interval to get the dimension

		if feat_type in ['IAT', 'FFT-IAT']:
			self.dim = dim - 1
			self.features, self.fids = _get_IAT(self.flows)
		elif feat_type in ['SIZE', 'FFT-SIZE']:
			self.dim = dim
			self.features, self.fids = _get_SIZE(self.flows)
		elif feat_type in ['IAT+SIZE','IAT_SIZE', 'FFT-IAT_SIZE']:
			self.dim = 2 * dim - 1
			self.features, self.fids = _get_IAT_SIZE(self.flows)
		elif feat_type in ['STATS']:
			self.dim = 10
			self.features, self.fids = _get_STATS(self.flows)
		elif feat_type in ['SAMP_NUM', 'FFT-SAMP_NUM']:
			self.dim = dim - 1
			# flow_durations = [_get_flow_duration(pkts) for fid, pkts in self.flows]
			# # To obtain different samp_features, you should change q_interval ((0, 1))
			# sampling_rate = _get_split_interval(flow_durations, q_interval=self.q_interval)
			self.features, self.fids = _get_SAMP_NUM(self.flows, self.sampling_rate)
		elif feat_type in ['SAMP_SIZE', 'FFT-SAMP_SIZE']:
			self.dim = dim - 1  # here the dim of "SAMP_SIZE" is dim -1, which equals to the dimension of 'SAMP_NUM'
			self.features, self.fids = _get_SAMP_SIZE(self.flows, self.sampling_rate)
		else:
			msg = f'feat_type ({feat_type}) is not correct! '
			raise ValueError(msg)

		lg.debug(f'self.dim: {self.dim}, feat_type: {feat_type}')
		if fft:
			self.features = _get_FFT_data(self.features, fft_bin=self.dim)
		else:
			# fix each flow to the same feature dimension (cut off the flow or append 0 to it)
			self.features = [v[:self.dim] if len(v) > self.dim else v + [0] * (self.dim - len(v)) for v in
			                 self.features]

		if header:
			_headers = _get_header_features(self.flows)
			h_dim = 8 + dim  # 8 TCP flags
			if fft:
				fft_headers = _get_FFT_data(_headers, fft_bin=h_dim)
				self.features = [h + f for h, f in zip(fft_headers, self.features)]
			else:
				# fix header dimension firstly
				headers = [h[:h_dim] if len(h) > h_dim else h + [0] * (h_dim - len(h)) for h in _headers]
				self.features = [h + f for h, f in zip(headers, self.features)]

		# change list to numpy array
		self.features = np.asarray(self.features, dtype=float)
		if self.verbose > 5:
			lg.debug(np.all(self.features >= 0))


def _subflows2featutes(flows, labels, dim=10, feat_type='IAT+SIZE', sampling_rate=0.1,
                       header=False, verbose=10):
	# extract features from each flow given feat_type
	""" Override the PCAP and have different sampling_rate for SAMP_SIZE

	Parameters
	----------
	flows
	labels
	dim
	feat_type
	sampling_rate
	header
	verbose

	Returns
	-------

	"""
	pp = New_PCAP(sampling_rate=sampling_rate)
	pp.flows = flows
	pp.labels = labels
	pp.flow2features(feat_type.upper(), fft=False, header=header, dim=dim)
	# out_file = f'{out_dir}/features-q_interval:{q_interval}.dat'
	# lg.debug('features+labels: ', out_file)
	# features = pp.features
	# labels = pp.labels
	# dump((features, labels), out_file)

	return pp.features, pp.labels

#
# class PCAP2FEATURES():
#
# 	def __init__(self, out_dir='', Xy_file=None,
# 	             feat_type='', header=False,
# 	             random_state=42, overwrite=False, verbose=0):
# 		self.out_dir = out_dir
# 		self.Xy_file = Xy_file
# 		self.feat_type = feat_type
# 		self.header = header
# 		self.verbose = 10
# 		self.random_state = random_state
# 		self.overwrite = overwrite
# 		if not os.path.exists(os.path.abspath(self.out_dir)): os.makedirs(self.out_dir)
#
# 	def get_path(self, datasets, original_dir, in_dir, direction):
# 		normal_files = []
# 		abnormal_files = []
# 		for _idx, _name in enumerate(datasets):
# 			normal_file, abnormal_file = _get_path(original_dir, in_dir, data_name=_name, direction=direction,
# 			                                       overwrite=self.overwrite)
# 			normal_files.append(normal_file)
# 			abnormal_files.append(abnormal_file)
#
# 		return normal_files, abnormal_files
#
# 	def flows2features(self, normal_files, abnormal_files, q_interval=0.9):
# 		is_same_duration = True
# 		if is_same_duration:
# 			self._flows2features(normal_files, abnormal_files, q_interval=q_interval)
# 		else:
# 			self._flows2features_seperate(normal_files, abnormal_files, q_interval=q_interval)
#
# 	def _flows2features(self, normal_files, abnormal_files, q_interval=0.9):
# 		lg.debug(f'normal_files: {normal_files}')
# 		lg.debug(f'abnormal_files: {abnormal_files}')
# 		durations = []
# 		normal_flows = []
# 		normal_labels = []
# 		for i, f in enumerate(normal_files):
# 			(flows, labels), load_time = time_func(load, f)
# 			normal_flows.extend(flows)
# 			lg.debug(f'i: {i}, load_time: {load_time} s.')
# 			normal_labels.extend([f'normal_{i}'] * len(labels))
# 			data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in flows]).reshape(-1, 1),
# 			          name=f'durations_{i}')
# 			durations.extend([_get_flow_duration(pkts) for fid, pkts in flows])
#
# 		# 1. get interval from all normal flows
# 		data_info(np.asarray(durations).reshape(-1, 1), name='durations')
# 		interval = _get_split_interval(durations, q_interval=q_interval)
# 		lg.debug(f'interval {interval} when q_interval: {q_interval}')
#
# 		abnormal_flows = []
# 		abnormal_labels = []
# 		for i, f in enumerate(abnormal_files):
# 			flows, labels = load(f)
# 			abnormal_flows.extend(flows)
# 			abnormal_labels.extend([f'abnormal_{i}'] * len(labels))
# 		lg.debug(f'fullflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
#
# 		# 2. flows2subflows
# 		flow_durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
# 		normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
# 		                                              flow_pkts_thres=2,
# 		                                              verbose=1)
# 		abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval, labels=abnormal_labels,
# 		                                                  flow_pkts_thres=2, verbose=1)
# 		lg.debug(f'subflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
#
# 		# 3. subflows2features
# 		num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
# 		# dim is for SIZE features
# 		dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
#
# 		if self.feat_type.startswith('SAMP'):
# 			X = {}
# 			y = {}
# 			for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
# 				# get sampling_rate on normal_flows first
# 				# lg.debug(f'np.quantile(flows_durations): {np.quantile(flow_durations, q=[0.1, 0.2, 0.3, 0.9, 0.95])}')
# 				sampling_rate = _get_split_interval(flow_durations, q_interval=q_samp)
# 				if sampling_rate <= 0.0: continue
# 				X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
# 				                                        feat_type=self.feat_type, sampling_rate=sampling_rate,
# 				                                        header=self.header, verbose=self.verbose)
#
# 				X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
# 				                                            feat_type=self.feat_type, sampling_rate=sampling_rate,
# 				                                            header=self.header,
# 				                                            verbose=self.verbose)
# 				lg.debug(
# 					f'q_samp: {q_samp}, subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
# 				self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
#
# 				X[q_samp] = np.concatenate([X_normal, X_abnormal], axis=0)
# 				y[q_samp] = np.concatenate([y_normal, y_abnormal], axis=0)
#
# 		else:
# 			X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
# 			                                        feat_type=self.feat_type, header=self.header, verbose=self.verbose)
# 			X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
# 			                                            feat_type=self.feat_type, header=self.header,
# 			                                            verbose=self.verbose)
# 			lg.debug(
# 				f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
# 			self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
# 			X = np.concatenate([X_normal, X_abnormal], axis=0)
# 			y = np.concatenate([y_normal, y_abnormal], axis=0)
# 		# self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
# 		dump((X, y), out_file=self.Xy_file)
# 		lg.debug(f'Xy_file: {self.Xy_file}')
#
# 	def _flows2features_seperate(self, normal_files, abnormal_files, q_interval=0.9):
# 			""" dataset1 and dataset2 use different interval and will get different dimension
# 				then append 0 to the smaller dimension to make both has the same dimension
#
# 			Parameters
# 			----------
# 			normal_files
# 			abnormal_files
# 			q_interval
#
# 			Returns
# 			-------
#
# 			"""
#
# 			lg.debug(f'normal_files: {normal_files}')
# 			lg.debug(f'abnormal_files: {abnormal_files}')
#
# 			X = []
# 			y = []
# 			for i, (f1, f2) in enumerate(zip(normal_files, abnormal_files)):
# 				(normal_flows, labels), load_time = time_func(load, f1)
# 				normal_labels = [f'normal_{i}'] * len(labels)
# 				lg.debug(f'i: {i}, load_time: {load_time} s.')
#
# 				# 1. get durations
# 				data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in normal_flows]).reshape(-1, 1),
# 				          name=f'durations_{i}')
# 				durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
#
# 				interval = _get_split_interval(durations, q_interval=q_interval)
# 				lg.debug(f'interval {interval} when q_interval: {q_interval}')
#
# 				# 2. flows2subflows
# 				normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
# 				                                              flow_pkts_thres=2,
# 				                                              verbose=1)
# 				# 3. subflows2features
# 				num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
# 				data_info(np.asarray(num_pkts).reshape(-1, 1), name='num_ptks_for_flows')
# 				dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
# 				lg.debug(f'i: {i}, dim: {dim}')
# 				X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
# 				                                        verbose=self.verbose)
# 				n_samples = 15000
# 				if len(y_normal) > n_samples:
# 					X_normal, y_normal = sklearn.utils.resample(X_normal, y_normal, n_samples=n_samples, replace=False,
# 					                                            random_state=42)
# 				else:
# 					X_normal, y_normal = sklearn.utils.resample(X_normal, y_normal, n_samples=60000, replace=True,
# 					                                            random_state=42)
# 				X.extend(X_normal.tolist())
# 				y.extend(y_normal)
#
# 				# for abnormal flows
# 				(abnormal_flows, labels), load_time = time_func(load, f2)
# 				abnormal_labels = [f'abnormal_{i}'] * len(labels)
# 				abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval,
# 				                                                  labels=abnormal_labels,
# 				                                                  flow_pkts_thres=2, verbose=1)
# 				X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
# 				                                            verbose=self.verbose)
# 				n_samples = 15000
# 				if len(y_abnormal) > n_samples:
# 					X_abnormal, y_abnormal = sklearn.utils.resample(X_abnormal, y_abnormal, n_samples=n_samples,
# 					                                                replace=False,
# 					                                                random_state=42)
# 				else:  #
# 					X_abnormal, y_abnormal = sklearn.utils.resample(X_abnormal, y_abnormal, n_samples=200, replace=True,
# 					                                                random_state=42)
#
# 				X.extend(X_abnormal.tolist())
# 				y.extend(y_abnormal)
# 				lg.debug(
# 					f'subflows (before sampling): normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
# 				lg.debug(
# 					f'after resampling: normal_labels: {Counter(y_normal)}, abnormal_labels: {Counter(y_abnormal)}')
# 			# break
# 			max_dim = max([len(v) for v in X])
# 			lg.debug(f'===max_dim: {max_dim}')
# 			new_X = []
# 			for v in X:
# 				v = v + (max_dim - len(v)) * [0]
# 				new_X.append(np.asarray(v, dtype=float))
#
# 			X = np.asarray(new_X, dtype=float)
# 			y = np.asarray(y, dtype=str)
# 			self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
# 			dump((X, y), out_file=self.Xy_file)
# 			lg.debug(f'Xy_file: {self.Xy_file}')
#
#


#
# def _get_path(original_dir, in_dir, data_name, overwrite=False, direction='src_dst'):
#     """
#
#     Parameters
#     ----------
#     in_dir
#     data_name
#     overwrite
#     direction: str
#         src_dst: use src + dst data
#         src: only user src data
#
#     Returns
#     -------
#
#     """
#     if 'UNB/CICIDS_2017/pc_' in data_name and 'Mon' not in data_name:
#         ##############################################################################################################
#         # step 1: get path
#         if data_name == 'UNB/CICIDS_2017/pc_192.168.10.5':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.pcap')
#             pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.csv')
#             if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.5'], direction=direction)
#                 # label_file
#                 in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
#                     'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
#                 out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
#                 merge_csvs(in_files, out_file)
#                 keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.5'], direction=direction, keep_original=True,
#                             verbose=10)
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#         elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.8':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.pcap')
#             pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.csv')
#             if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.8'], direction=direction)
#                 # label_file
#                 in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
#                     'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
#                 out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
#                 merge_csvs(in_files, out_file)
#                 keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.8'], direction=direction, keep_original=True,
#                             verbose=10)
#
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#         elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.9':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.9.pcap')
#             pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.9.csv')
#             if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.9'], direction=direction)
#                 # label_file
#                 in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
#                     'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
#                 out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
#                 merge_csvs(in_files, out_file)
#                 keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.9'], direction=direction, keep_original=True,
#                             verbose=10)
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#
#         elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.14':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.14.pcap')
#             pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.14.csv')
#             if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.14'], direction=direction)
#                 # label_file
#                 in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
#                     'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
#                 out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
#                 merge_csvs(in_files, out_file)
#                 keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.14'], direction=direction, keep_original=True,
#                             verbose=10)
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#         elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.15':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.15.pcap')
#             pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.15.csv')
#             if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.15'], direction=direction)
#                 # label_file
#                 in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
#                     'Friday-WorkingHours-Morning.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
#                     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
#                 out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
#                 merge_csvs(in_files, out_file)
#                 keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.15'], direction=direction, keep_original=True,
#                             verbose=10)
#
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#         else:
#             msg = f'{data_name} does not found.'
#             raise ValueError(msg)
#
#         ##############################################################################################################
#         # step 2:  pcap 2 flows
#         normal_file = os.path.dirname(pth_pcap_mixed) + '/normal_flows_labels.dat'
#         abnormal_file = os.path.dirname(pth_pcap_mixed) + '/abnormal_flows_labels.dat'
#         if overwrite:
#             if os.path.exists(normal_file): os.remove(normal_file)
#             if os.path.exists(abnormal_file): os.remove(abnormal_file)
#         if not os.path.exists(normal_file) or not os.path.exists(abnormal_file):
#             normal_flows, normal_labels, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pth_pcap_mixed,
#                                                                                            label_file=pth_labels_mixed)
#
#             dump((normal_flows, normal_labels), out_file=normal_file)
#             dump((abnormal_flows, abnormal_labels), out_file=abnormal_file)
#
#     else:
#         ##############################################################################################################
#         # step 1: get path
#         if data_name == 'UCHI/IOT_2019/smtv_10.42.0.1':
#             # # normal and abormal are independent
#             #  editcap -c 500000 merged.pcap merged.pcap
#             pth_normal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.1.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119-abnormal.pcap')
#             # pth_labels_normal, pth_labels_abnormal = None, None
#
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 merged_pcap = pth.join(in_dir, direction, data_name, 'merged.pcap')
#                 if not os.path.exists(merged_pcap):
#                     pcap_files = [pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps', v)
#                                   for v in os.listdir(
#                             pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps')) if
#                                   not v.startswith('.')]
#                     merge_pcaps(in_files=pcap_files, out_file=merged_pcap)
#
#                 # 10.42.0.1 date is from the whole megerd pcap
#                 keep_ip(merged_pcap, out_file=pth_normal, kept_ips=['10.42.0.1'], direction=direction)
#
#                 merged_pcap = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119_00000_20190927224625.pcap')
#                 keep_ip(merged_pcap, out_file=pth_abnormal, kept_ips=['10.42.0.119'], direction=direction)
#                 # copyfile(idle_pcap, pth_normal)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2019/smtv_10.42.0.119':
#             # # normal and abormal are independent
#             #  editcap -c 500000 merged.pcap merged.pcap
#             pth_normal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119.pcap', )
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.1.pcap')
#             # pth_labels_normal, pth_labels_abnormal = None, None
#
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 merged_pcap = pth.join(in_dir, direction, data_name, 'merged_00000_20190927182134.pcap')
#                 if not os.path.exists(merged_pcap):
#                     pcap_files = [pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps', v)
#                                   for v in os.listdir(
#                             pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps')) if
#                                   not v.startswith('.')]
#                     merge_pcaps(in_files=pcap_files, out_file=merged_pcap)
#                 keep_ip(merged_pcap, out_file=pth_normal, kept_ips=['10.42.0.119'], direction=direction)
#                 keep_ip(merged_pcap, out_file=pth_abnormal, kept_ips=['10.42.0.1'], direction=direction)
#                 # copyfile(idle_pcap, pth_normal)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#
#         elif data_name == 'OCS1/IOT_2018/pc_192.168.0.13':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'pc_192.168.0.13-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_192.168.0.13-anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'OCS/IOT_2018', 'pcaps',
#                                    'benign-dec.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.0.13'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'OCS/IOT_2018', 'pcaps',
#                                    'mirai-udpflooding-2-dec.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.0.13'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name,
#                                   '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU/IOT_2017/pc_192.168.1.196':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name,
#                                   '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.196'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.195':
#             """
#             normal_traffic:
#                 https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
#                 https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/
#             """
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.196':
#             """
#                         normal_traffic:
#                             https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
#                         """
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.196'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.195':
#             """
#             normal_traffic:
#                 https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
#                 https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/
#             """
#             ## editcap -c 500000 2017-05-02_kali.pcap 2017-05-02_kali.pcap
#
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-05-02_kali_00000_20170502082205.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.191'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.196':
#             """
#             normal_traffic:
#                 https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
#             """
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_abnormal.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 # in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-05-02_kali_00000_20170502072205.pcap')   # for CALUMENT
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    '2017-05-02_kali_00000_20170502082205.pcap')  # for NEON
#
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.191'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'CTU/IOT_2017',
#                                    'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.196'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2019/pc_202.171.168.50':
#             # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
#             # file_name = 'samplepoint-F_201912071400_00000_20191207000000.pcap'
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name,
#                                   '201912071400-pc_202.171.168.50_normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '201912071400-pc_202.4.27.109_anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 if direction == 'src_dst':
#                     in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
#                                        'samplepoint-F_201912071400-src_dst_202.171.168.50-5000000.pcap')
#                     # editcap -c 5000000 samplepoint-F_201912071400-src_dst_202.171.168.50.pcap samplepoint-F_201912071400-src_dst_202.171.168.50-.pcap
#                 else:
#                     in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
#                                        'samplepoint-F_201912071400-src_dst_202.171.168.50.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['202.171.168.50'], direction=direction)
#
#                 # in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
#                 #                    'samplepoint-F_201912071400_00000_20191207000000.pcap')
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
#                                    'samplepoint-F_201912071400-src_dst_202.4.27.109.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['202.4.27.109'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2020/pc_203.78.7.165':
#             # normal and abormal are independent
#             # editcap -c 30000000 samplepoint-F_202007011400.pcap samplepoint-F_202007011400.pcap
#             # tshark -r samplepoint-F_202007011400.pcap -w 202007011400-pc_203.78.7.165.pcap ip.addr==203.78.7.165
#             pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.7.165.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '202007011400-pc_185.8.54.240.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400_00000_20200701010000.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.7.165'], direction=direction)
#
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400_00000_20200701010000.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['185.8.54.240'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2020/pc_203.78.4.32':
#             pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.4.32.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '202007011400-pc_202.75.33.114.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.4.32'], direction=direction)
#
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400.pcap-src_dst_202.75.33.114.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['202.75.33.114'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2020/pc_203.78.4.32-2':
#
#             pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.4.32.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '202007011400-pc_203.78.8.151.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.4.32'], direction=direction)
#
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['203.78.8.151'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2020/pc_203.78.7.165-2':
#             pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.7.165.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     '202007011400-pc_203.78.8.151.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400-src_dst_203.78.7.165.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.7.165'], direction=direction)
#
#                 in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
#                                    'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['203.78.8.151'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2019/pc_203.78.4.32':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.4.32.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     '202007011400-srcIP_203.78.7.165.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2019/pc_222.117.214.171':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.7.165.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     '202007011400-srcIP_222.117.214.171.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2019/pc_101.27.14.204':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.7.165.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     '202007011400-srcIP_101.27.14.204.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MAWI/WIDE_2019/pc_18.178.219.109':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.4.32.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     '202007011400-srcIP_18.178.219.109.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2019/ghome_192.168.143.20':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'ghome_192.168.143.20-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'ghome_192.168.143.20-anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/ghome_192.168.143.20',
#                                    'fridge_cam_sound_ghome_2daysactiv-ghome_normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.20'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/ghome_192.168.143.20',
#                                    'fridge_cam_sound_ghome_2daysactiv-ghome_abnormal.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.20'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2019/scam_192.168.143.42':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'scam_192.168.143.42-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     'scam_192.168.143.42-anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/scam_192.168.143.42',
#                                    'fridge_cam_sound_ghome_2daysactiv-scam_normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.42'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/scam_192.168.143.42',
#                                    'fridge_cam_sound_ghome_2daysactiv-scam_abnormal.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.42'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2019/sfrig_192.168.143.43':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'sfrig_192.168.143.43-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     'sfrig_192.168.143.43-anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/sfrig_192.168.143.43',
#                                    'fridge_cam_sound_ghome_2daysactiv-sfrig_normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.43'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/sfrig_192.168.143.43',
#                                    'fridge_cam_sound_ghome_2daysactiv-sfrig_abnormal.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.43'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2019/bstch_192.168.143.48':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'bstch_192.168.143.48-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name,
#                                     'bstch_192.168.143.48-anomaly.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/bstch_192.168.143.48',
#                                    'fridge_cam_sound_ghome_2daysactiv-bstch_normal.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.48'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'UCHI/IOT_2019/bstch_192.168.143.48',
#                                    'fridge_cam_sound_ghome_2daysactiv-bstch_abnormal.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.48'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         # elif data_name == 'UCHI/IOT_2019/pc_192.168.143.43':
#         #     # normal and abormal are independent
#         #     # 'idle'
#         #     # 'iotlab_open_shut_fridge_192.168.143.43/open_shut'
#         #     pth_normal = pth.join(in_dir, data_name, 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap')
#         #     pth_abnormal = pth.join(in_dir, data_name,
#         #                             'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap')
#         #     pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2020/aecho_192.168.143.74':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'idle-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'shop-anomaly.pcap')
#             # pth_abnormal = pth.join(in_dir, direction, data_name)   # directory
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
#                                           out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
#                                           kept_ips=['192.168.143.74'], direction=direction)
#                                   for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
#                                   not v.startswith('.')]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_normal)
#                 # abnormal
#                 # abnormal_file = pth.join(original_dir, data_name, 'echo_song.pcap')
#                 # Can not use the whole abnormal pcap directly because when we split it to subpcap,
#                 # one flow will split to multi-flows.
#                 # pth_abnormal = keep_ip(abnormal_file, out_file=pth_abnormal, kept_ips=['192.168.143.74'],
#                 #                        direction=direction)
#                 activity = 'shop'
#                 whole_abnormal = pth.join(original_dir, data_name, f'echo_{activity}.pcap')
#                 num = split_by_activity(whole_abnormal, out_dir=os.path.dirname(whole_abnormal), activity=activity)
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, data_name, v),
#                                           out_file=pth.join(os.path.dirname(idle_pcap), v),
#                                           kept_ips=['192.168.143.74'], direction=direction)
#                                   for v in [f'{activity}/capture{i}.seq/deeplens_{activity}_{i}.' \
#                                             f'pcap' for i in range(num)]
#                                   ]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_abnormal)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2020/sfrig_192.168.143.43':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'open_shut.pcap')
#             # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
#                                           out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
#                                           kept_ips=['192.168.143.43'], direction=direction)
#                                   for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
#                                   not v.startswith('.')]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_normal)
#
#                 # abnormal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, data_name, v),
#                                           out_file=pth.join(os.path.dirname(idle_pcap), v),
#                                           kept_ips=['192.168.143.43'], direction=direction)
#                                   for v in [f'open_shut/capture{i}.seq/deeplens_open_shut_fridge_batch_{i}.' \
#                                             f'pcap_filtered.pcap' for i in range(9)]
#                                   ]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_abnormal)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2020/wshr_192.168.143.100':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'open_wshr.pcap')
#             # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
#                                           out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
#                                           kept_ips=['192.168.143.100'], direction=direction)
#                                   for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
#                                   not v.startswith('.')]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_normal)
#
#                 # abnormal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, data_name, v),
#                                           out_file=pth.join(os.path.dirname(idle_pcap), v),
#                                           kept_ips=['192.168.143.100'], direction=direction)
#                                   for v in [f'open_wshr/capture{i}.seq/deeplens_open_washer_{i}.' \
#                                             f'pcap' for i in range(31)]
#                                   ]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_abnormal)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'open_dwshr.pcap')
#             # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
#                                           out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
#                                           kept_ips=['192.168.143.76'], direction=direction)
#                                   for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
#                                   not v.startswith('.') and 'pcap' in v]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_normal)
#
#                 # abnormal
#                 idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
#                 if not os.path.exists(idle_pcap):
#                     idle_files = [keep_ip(pth.join(original_dir, data_name, v),
#                                           out_file=pth.join(os.path.dirname(idle_pcap), v),
#                                           kept_ips=['192.168.143.76'], direction=direction)
#                                   for v in [f'open_dwshr/capture{i}.seq/deeplens_open_dishwasher_{i}.' \
#                                             f'pcap_filtered.pcap' for i in range(31)]
#                                   ]
#                     merge_pcaps(in_files=idle_files, out_file=idle_pcap)
#                 copyfile(idle_pcap, pth_abnormal)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'UCHI/IOT_2020/ghome_192.168.143.20':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'WRCCDC/2020-03-20':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, 'wrccdc.2020-03-20.174351000000002-172.16.16.30-normal.pcap')
#             # pth_abnormal = pth.join(in_dir, data_name,
#             #                         'wrccdc.2020-03-20.174351000000002-172.16.16.16.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     'wrccdc.2020-03-20.174351000000002-10.183.250.172-abnormal.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#         elif data_name == 'DEFCON/ctf26':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, data_name, 'DEFCON26ctf_packet_captures-src_10.0.0.2-normal.pcap')
#             pth_abnormal = pth.join(in_dir, data_name,
#                                     'DEFCON26ctf_packet_captures-src_10.13.37.23-abnormal.pcap')
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'ISTS/ISTS_2015':
#             # normal and abormal are independent
#             # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425741051-src_10.128.0.13-normal.pcap')
#             # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425823409-src_10.2.1.80.pcap')
#             # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425824560-src_129.21.3.17.pcap')
#             # pth_normal = pth.join(in_dir, data_name,
#             #                       'snort.log-merged-srcIP_10.128.0.13-10.0.1.51-10.0.1.4-10.2.12.40.pcap')
#             #
#             # pth_abnormal = pth.join(in_dir, data_name,
#             #                         'snort.log-merged-srcIP_10.2.12.50.pcap')
#
#             pth_normal = pth.join(in_dir, direction, data_name, 'snort.log-merged-3pcaps-normal.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'snort.log.1425824164.pcap')
#             if not pth.exists(pth_normal) or not pth.exists(pth_abnormal):
#                 in_files = [
#                     'snort.log.1425741002.pcap',
#                     'snort.log.1425741051.pcap',
#                     'snort.log.1425823409.pcap',
#                     # 'snort.log.1425842738.pcap',
#                     # 'snort.log.1425824560.pcap',
#
#                 ]
#                 # in_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425824164.pcap' # for abnormal datasets
#                 in_files = [os.path.join(original_dir, data_name, v) for v in in_files]
#                 out_file = os.path.join(in_dir, direction, data_name, 'snort.log-merged-3pcaps.pcap')
#                 merge_pcaps(in_files, out_file)
#                 copyfile(out_file, pth_normal)
#                 in_file = pth.join(original_dir, data_name, 'snort.log.1425824164.pcap')
#                 copyfile(in_file, pth_abnormal)
#             # pth_abnormal = pth.join(in_dir, data_name, 'snort.log-merged-srcIP_10.2.4.30.pcap')
#             # if not pth.exists(pth_abnormal):
#             #     out_file = keep_ip(pth_normal, out_file=pth_abnormal, kept_ips=['10.2.4.30'])
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'MACCDC/MACCDC_2012/pc_192.168.202.79':
#             # normal and abormal are independent
#             # pth_normal = pth.join(in_dir, data_name, 'maccdc2012_00000-srcIP_192.168.229.153.pcap')
#             # the result does beat OCSVM.
#             pth_normal = pth.join(in_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.79.pcap')
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.76.pcap')
#             if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
#                 # normal
#                 in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.202.79'], direction=direction)
#                 # abnormal
#                 in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.202.76'], direction=direction)
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         elif data_name == 'CTU_IOT23/CTU-IoT-Malware-Capture-7-1':
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, data_name, '2018-07-20-17-31-20-192.168.100.108.pcap')
#             pth_labels_mixed = pth.join(in_dir, data_name,
#                                         'CTU-IoT-Malware-Capture-7-1-conn.log.labeled.txt.csv-src_192.168.100.108.csv')
#
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#         elif data_name == 'UNB/CICIDS_2017/Mon-pc_192.168.10.5':
#             # normal and abormal are independent
#             pth_normal = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.pcap')
#             if not os.path.exists(pth_normal):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Monday/Monday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.10.5'], direction=direction)
#
#             # normal and abormal are mixed together
#             pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.pcap')
#             if not os.path.exists(pth_abnormal):
#                 in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Monday/Monday-WorkingHours.pcap')
#                 keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.10.8'], direction=direction)
#
#             pth_labels_normal, pth_labels_abnormal = None, None
#
#         else:
#             print('debug')
#             data_name = 'DEMO_IDS/DS-srcIP_192.168.10.5'
#             # normal and abormal are mixed together
#             pth_pcap_mixed = pth.join(in_dir, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
#             pth_labels_mixed = pth.join(in_dir, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')
#
#             pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
#
#         ##############################################################################################################
#         # step 2: pcap 2 flows
#         normal_file = os.path.dirname(pth_normal) + '/normal_flows_labels.dat'
#         abnormal_file = os.path.dirname(pth_normal) + '/abnormal_flows_labels.dat'
#         if overwrite:
#             if os.path.exists(normal_file): os.remove(normal_file)
#             if os.path.exists(abnormal_file): os.remove(abnormal_file)
#         if not os.path.exists(normal_file) or not os.path.exists(abnormal_file):
#             normal_flows, normal_labels, _, _ = _pcap2fullflows(pcap_file=pth_normal,
#                                                                 label_file=None, label='normal')
#             _, _, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pth_abnormal,
#                                                                     label_file=None, label='abnormal')
#
#             dump((normal_flows, normal_labels), out_file=normal_file)
#             dump((abnormal_flows, abnormal_labels), out_file=abnormal_file)
#
#     print(f'normal_file: {normal_file}, exists: {pth.exists(normal_file)}')
#     print(f'abnormal_file: {abnormal_file}, exists: {pth.exists(abnormal_file)}')
#
#     return normal_file, abnormal_file
#
#
