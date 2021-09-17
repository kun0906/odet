""" Base class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os

import numpy as np
from loguru import logger as lg

from odet.pparser.parser import _get_flow_duration, _get_split_interval, _flows2subflows, _get_IAT, _get_SIZE, \
	_get_IAT_SIZE, _get_STATS, _get_SAMP_NUM, _get_SAMP_SIZE, _get_FFT_data, _pcap2flows
from odet.utils.tool import check_path, dump, timer, remove_file, load


def _get_SAMP(flows, name='SAMP_NUM', dim=None, header=False, header_dim=None):
	qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
	flow_durations = [_get_flow_duration(pkts) for fid, pkts in flows]
	features_mp = {}
	fids_mp = {}
	for q in qs:
		sampling_rate_ = np.quantile(flow_durations, q=q)
		if name in ['SAMP_NUM', 'FFT_SAMP_NUM']:
			features, fids = _get_SAMP_NUM(flows, sampling_rate_)
		elif name in ['SAMP_SIZE', 'FFT_SAMP_SIZE']:
			features, fids = _get_SAMP_SIZE(flows, sampling_rate_)

		new_dim = dim
		if 'FFT' in name:
			features = _get_FFT_data(features, fft_bin=new_dim, fft_part='real')
			if header:
				header_features, header_fids = _get_header(flows)
				header_features = _get_FFT_data(header_features, fft_bin=header_dim,
				                                fft_part='real')  # 8 is the number of tcp_flg
				features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header
		else:
			features = _fix_data(features, new_dim)
			if header:
				header_features, header_fids = _get_header(flows)
				header_features = _fix_data(header_features, header_dim)
				features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header

		features_mp[q] = (features, fids, sampling_rate_)
		fids_mp[q] = (fids)
	return features_mp, fids_mp


class Base:

	def __init__(self, Xy_file=None, feature_name='IAT+SIZE'):
		self.Xy_file = Xy_file
		self.feature_name = feature_name

	def pcap2flows(self, pcap_file=None):
		return _pcap2flows(pcap_file, flow_pkts_thres=2)

	def flow2subflows(self, flows=None, interval=None, labels=None):
		return _flows2subflows(flows, interval=interval, labels=labels)

	def flow2features(self, flows=None, name='IAT+SIZE', dim=10, header=False, header_dim=None):

		if name in ['IAT', 'FFT_IAT', 'SIZE', 'FFT_SIZE', 'IAT+SIZE', 'FFT_IAT+SIZE']:
			if name in ['IAT', 'FFT_IAT']:
				features, fids = _get_IAT(flows)
				new_dim = dim - 1
			elif name in ['SIZE', 'FFT_SIZE']:
				features, fids = _get_SIZE(flows)
				new_dim = dim
			elif name in ['IAT+SIZE', 'FFT_IAT+SIZE']:
				features, fids = _get_IAT_SIZE(flows)
				new_dim = 2 * dim - 1

			if 'FFT' in name:
				features = _get_FFT_data(features, fft_bin=new_dim, fft_part='real')
				if header:
					header_features, header_fids = _get_header(flows)
					header_features = _get_FFT_data(header_features, fft_bin=header_dim,
					                                fft_part='real')  # 8 is the number of tcp_flg
					features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header
				else:
					pass
			else:  # without FFT
				features = _fix_data(features, new_dim)
				if header:
					header_features, header_fids = _get_header(flows)
					header_features = _fix_data(header_features, header_dim)
					features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header
				else:
					pass
			return features, fids

		elif name in ['STATS', 'FFT_STATS']:
			features, fids = _get_STATS(flows)
			new_dim = 10
			if 'FFT' in name:
				features = _get_FFT_data(features, fft_bin=new_dim, fft_part='real')
				if header:
					header_features, header_fids = _get_header(flows)
					header_features = _get_FFT_data(header_features, fft_bin=header_dim,
					                                fft_part='real')  # 8 is the number of tcp_flg
					features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header
			else:
				features = _fix_data(features, new_dim)
				if header:
					header_features, header_fids = _get_header(flows)
					header_features = _fix_data(header_features, header_dim)
					features = np.concatenate([features, header_features], axis=1)  # concatanate feature and header
				else:
					pass
			return features, fids
		elif name in ['SAMP_NUM', 'FFT_SAMP_NUM', 'SAMP_SIZE', 'FFT_SAMP_SIZE']:
			if name in ['SAMP_NUM', 'FFT_SAMP_NUM']:
				new_dim = dim
				features, fids = _get_SAMP(flows, name=name, dim=new_dim, header=header, header_dim=header_dim)
			elif name in ['SAMP_SIZE', 'FFT_SAMP_SIZE']:
				new_dim = dim
				features, fids = _get_SAMP(flows, name=name, dim=new_dim, header=header, header_dim=header_dim)
		else:
			msg = f'{name}'
			raise NotImplementedError(msg)

		return features, fids

	def fix_feature(self, features, dim=None):
		# fix each flow to the same feature dimension (cut off the flow or append 0 to it)
		features = [v[:dim] if len(v) > dim else v + [0] * (dim - len(v)) for v in features]
		return np.asarray(features)

	def generate(self):
		q_interval = 0.9
		# pcap to flows
		flows = self.pcap2flows(self.pcap_file)

		# flows to subflow
		labels = [1] * len(flows)
		durations = [_get_flow_duration(pkts) for fid, pkts in flows]
		interval = _get_split_interval(durations, q_interval=q_interval)
		subflows, labels = self.flow2subflows(flows, interval=interval, labels=labels)

		# get dimension
		normal_flows = subflows
		num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
		dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
		lg.info(f'dim={dim}')

		# flows to features
		features, fids = self.flow2features(subflows, name=self.feature_name)

		# fixed the feature size
		features = self.fix_feature(features, dim=dim)

		self.X = features
		self.y = np.asarray([0] * len(features))

		# save data to disk
		check_path(os.path.dirname(self.Xy_file))
		dump((self.X, self.y), out_file=self.Xy_file)
		msg = 'Base.generate()'
		raise NotImplementedError(msg)

	def _generate_features(self, normal_flows, abnormal_flows):
		# step 3: flows to features.
		# only on normal flows
		normal_flow_lengths = [len(pkts) for fid, pkts in normal_flows]
		qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
		normal_lengths_stat = np.quantile(normal_flow_lengths, q=qs)
		lg.debug(f'normal_lengths_stat: {normal_lengths_stat}, where q = {qs}')
		self.dim = int(np.floor(np.quantile(normal_flow_lengths, self.q_flow_dur)))
		lg.info(f'dim(SIZE) = {self.dim}')

		self.X = []
		self.y = []

		if self.header:
			header_features, header_fids = _get_header(normal_flows)
			header_dim = int(np.quantile([len(v) for v in header_features], q=self.q_flow_dur))
			lg.info(f'header_dim: {header_dim}')
		else:
			header_dim = None

		if 'SAMP' in self.feature_name:
			normal_features, normal_fids = self.flow2features(normal_flows, name=self.feature_name, dim=self.dim,
			                                                  header=self.header, header_dim=header_dim)
			abnormal_features, abnormal_fids = self.flow2features(abnormal_flows, name=self.feature_name, dim=self.dim,
			                                                      header=self.header, header_dim=header_dim)

			for q in normal_features.keys():
				X_ = list(normal_features[q][0])  # (features, fid, sampling_rate_)
				y_ = [0] * len(normal_features[q][0])
				X_.extend(list(abnormal_features[q][0]))
				y_.extend([1] * len(abnormal_features[q][0]))
				self.X.append(np.asarray(X_))
				self.y.append(np.asarray(y_))

			# save data to disk
			check_path(self.Xy_file)
			meta = {'X': self.X, 'y': self.y,
			        'normal_flow_lengths': (normal_flow_lengths, normal_lengths_stat),
			        'dim': self.dim, 'q_flow_dur': self.q_flow_dur}
			dump(meta, out_file=self.Xy_file)
			# save feature data as csv
			csv_file = os.path.splitext(self.Xy_file)[0] + '.csv'
		# np.savetxt(csv_file, np.concatenate([self.X, self.y[..., np.newaxis]], axis=1), delimiter=',')
		else:
			for flows, label in zip([normal_flows, abnormal_flows], [0, 1]):
				features, fids = self.flow2features(flows, name=self.feature_name, dim=self.dim,
				                                    header=self.header, header_dim=header_dim)
				self.X.extend(features)
				self.y.extend([label] * len(features))

			# save data to disk
			check_path(self.Xy_file)
			self.X = np.asarray(self.X)
			self.y = np.asarray(self.y)
			meta = {'X': self.X, 'y': self.y,
			        'normal_flow_lengths': (normal_flow_lengths, normal_lengths_stat),
			        'dim': self.dim, 'q_flow_dur': self.q_flow_dur}
			dump(meta, out_file=self.Xy_file)
			# save feature data as csv
			csv_file = os.path.splitext(self.Xy_file)[0] + '.csv'
			np.savetxt(csv_file, np.concatenate([self.X, self.y[..., np.newaxis]], axis=1), delimiter=',')
		return meta

	@timer
	def _generate_flows(self):
		self.subflows_file = os.path.join(self.out_dir, 'normal_abnormal_subflows.dat')
		remove_file(self.subflows_file, self.overwrite)
		if os.path.exists(self.subflows_file):
			return load(self.subflows_file)

		# step 2: extract flows from pcap
		##############################################################################################
		meta = load(self.orig_flows)
		normal_flows, abnormal_flows = meta['normal_flows'], meta['abnormal_flows']
		lg.debug(f'original normal flows: {len(normal_flows)} and abnormal flows: {len(abnormal_flows)}')
		qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
		len_stat = np.quantile([len(pkts) for f, pkts in normal_flows], q=qs)
		lg.debug(f'flows: {len(normal_flows)}, length statistic: {len_stat}, when q = {qs}')
		meta = {'flows': normal_flows, 'len_stat': (len_stat, qs),
		        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows}
		dump(meta, out_file=os.path.join(self.out_dir, 'normal_abnormal_flows.dat'))

		# step 2.2. only get normal flows durations
		self.flows_durations = [_get_flow_duration(pkts) for (fids, pkts) in normal_flows]
		normal_durations_stat = np.quantile(self.flows_durations, q=qs)
		lg.debug(f'normal_durations_stat: {normal_durations_stat}')
		self.subflow_interval = np.quantile(self.flows_durations, q=self.q_flow_dur)  # median  of flow_durations
		lg.debug(f'---subflow_interval: {self.subflow_interval}, q_flow_dur: {self.q_flow_dur}')
		# step 2.3 get subflows
		normal_flows, _ = _flows2subflows(normal_flows, interval=self.subflow_interval,
		                                  labels=['0'] * len(normal_flows))
		abnormal_flows, _ = _flows2subflows(abnormal_flows, interval=self.subflow_interval,
		                                    labels=['1'] * len(abnormal_flows))
		lg.debug(f'normal_flows: {len(normal_flows)}, and abnormal_flows: {len(abnormal_flows)} '
		         f'with interval: {self.subflow_interval} and q: {self.q_flow_dur}')
		meta = {'normal_flows_durations': self.flows_durations, 'normal_durations_stat': (normal_durations_stat, qs),
		        'subflow_interval': self.subflow_interval, 'q_flow_dur': self.q_flow_dur,
		        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows}
		dump(meta, out_file=self.subflows_file)

		# only return subflows
		return meta


def _fix_data(features, feat_dim):
	""" Fix data by appending '0' or cutting off

	Parameters
	----------
	features

	feat_dim: int
		the fixed dimension of features

	Returns
	-------
	fixed_features:
		the fixed features
	"""
	fixed_features = []
	for feat in features:
		feat = list(feat)
		if len(feat) > feat_dim:
			feat = feat[:feat_dim]
		else:
			feat += [0] * (feat_dim - len(feat))

		fixed_features.append(np.asarray(feat, dtype=float))

	return np.asarray(fixed_features, dtype=float)


def _get_fft_data(features, fft_bin='', fft_part='real', feat_set='fft_iat'):
	""" Do fft transform of features

	Parameters
	----------
	features: features

	fft_bin: int
		the dimension of transformed features
	fft_part: str
		'real' or 'real+imaginary' transformation

	feat_set: str

	Returns
	-------
	fft_features:
		transformed fft features
	"""
	if fft_part == 'real':  # default
		fft_features = [np.real(np.fft.fft(v, n=fft_bin)) for v in features]

	elif fft_part == 'real+imaginary':
		fft_features = []
		for i, v in enumerate(features):
			complex_v = np.fft.fft(v, fft_bin)
			if i == 0:
				print(f'dimension of the real part: {len(np.real(complex_v))}, '
				      f'dimension of the imaginary part: {len(np.imag(complex_v))}')
			v = np.concatenate([np.real(complex_v), np.imag(complex_v)], axis=np.newaxis)
			fft_features.append(v)

	else:
		print(f'fft_part: {fft_part} is not correct, please modify it and retry')
		return -1

	return np.asarray(fft_features, dtype=float)


def _get_header(flows):
	features = []
	fids = []
	for fid, pkts in flows:  # fid, IAT, pkt_len
		features.append(get_header_features(pkts))
		fids.append(fid)
	return features, fids


def _get_header_from_flows(flows):
	# convert Unix timestamp arrival times into interpacket intervals
	flows = [(fid, np.diff(times), pkts) for (fid, times, pkts) in flows]  # No need to use sizes[1:]
	features_header = []
	for fid, times, pkts in flows:  # fid, IAT, pkt_len
		features_header.append((fid, get_header_features(pkts)))  # (fid, np.array())

	return features_header


def parse_tcp_flgs(tcp_flgs):
	# flags = {
	#     'F': 'FIN',
	#     'S': 'SYN',
	#     'R': 'RST',
	#     'P': 'PSH',
	#     'A': 'ACK',
	#     'U': 'URG',
	#     'E': 'ECE',
	#     'C': 'CWR',
	# }
	flgs = {
		'F': 0,
		'S': 0,
		'R': 0,
		'P': 0,
		'A': 0,
		'U': 0,
		'E': 0,
		'C': 0,
	}
	# flags = sorted(flags.items(), key=lambda x:x[0], reverse=True)
	# flags = OrderedDict(flags.items())
	# flg_lst = [0]*len(flags)
	# [flags[x] for x in p.sprintf('%TCP.flags%')]
	# ['SYN', 'ACK']
	for flg in tcp_flgs:
		if flg in flgs.keys():
			flgs[flg] += 1

	return list(flgs.values())


def get_header_features(pkts):
	features = []
	flgs_lst = np.zeros((8, 1))
	for i, pkt in enumerate(pkts):
		if pkt.payload.proto == 6:  # tcp
			flgs_lst += np.asarray(parse_tcp_flgs(pkt.payload.payload.flags)).reshape(-1, 1)  # parses tcp.flgs
		# elif pkt.payload.proto ==17: # udp
		#   pass
		features.append(pkt.payload.ttl)
	# features.append(pkt.payload.payload.dport)    # add dport will get 100% accuracy

	flgs_lst = list(flgs_lst.flatten())
	flgs_lst.extend(features)  # add ttl to the end of flgs.
	features = flgs_lst
	return features


def _flows_to_iats_sizes(flows, feat_set='iat', verbose=True):
	'''Get IATs features from flows

	   Arguments:
		 flows (list) = representation returned from read_pcap

	   Returns:
		 features (list) = [(fid, IATs)]
	'''
	# convert Unix timestamp arrival times into interpacket intervals
	# calculate IATs
	# the flows should be a deep copy of original flows. copy.deepcopy(flows)
	# flows = copy.deepcopy(flows_lst)      # may takes too much time
	if verbose:  # for verifying the code
		cnt = 3  # only show 3 flows
		cnt_1 = cnt
		flg = False
		for i, (fid, times, pkts) in enumerate(flows):  # flows is a list [(fid, times, pkts)]
			sizes = [len(pkt) for pkt in pkts]
			iats = np.diff(times)
			if (0 in iats) or (len(iats[iats == 0])):  # if two packets have the same timestamp?
				l = min(len(times), 10)
				# only print part of data
				print(f'i: {i}, 0 in np.diff(times): fid: {fid}, times (part of times to display): {times[:l]}, '
				      f'sizes: {sizes[:l]}, one reason is that retransmitted packets have the same time '
				      f'in wireshark, please check the pcap')
				cnt += 1
				if cnt > 3:
					flg = True
			if sum(iats) == 0:  # flow's duration is 0.0. Is it possible?
				# One reason is that the flow only have two kinds of packets:
				# one is the sent packet, the rest of them  is the retransmitted packets which has the same time
				# to the sent packet in wireshark, please check
				print(f'i: {i}, sum(np.diff(times)) == 0:  fid: {fid}, times: {times}, sizes: {sizes}')
				cnt_1 += 1
				if cnt_1 > 3:
					flg = True
			if flg:
				break

	if feat_set == 'iat':
		features = [(fid, np.asarray(list(np.diff(times)))) for (fid, times, pkts) in
		            flows]  # (fid, np.array())
	elif feat_set == 'size':
		features = [(fid, np.asarray([len(pkt) for pkt in pkts])) for (fid, times, pkts) in
		            flows]  # (fid, np.array())
	elif feat_set == 'iat_size':
		# features = [(fid, np.asarray(list(np.diff(times)) + [len(pkt) for pkt in pkts])) for (fid, times, pkts) in
		#             flows]  # (fid, np.array())
		features = []
		for (fid, times, pkts) in flows:
			feat = []
			feat_1 = list(np.diff(times))
			feat_2 = [len(pkt) for pkt in pkts]
			for i in range(len(times) - 1):
				feat.extend([feat_1[i], feat_2[i]])
			feat.append(feat_2[-1])
			features.append((fid, np.asarray(feat, dtype=float)))

	else:
		raise NotImplementedError(
			f'{feat_set} is not implemented, {os.path.relpath(_flows_to_iats_sizes.__code__.co_filename)}' \
			f'at line {_flows_to_iats_sizes.__code__.co_firstlineno}\'')

	return features


def sampling_packets(flow, sampling_type='rate', sampling=5, sampling_feature='samp_num', random_state=42):
	"""

	:param flow:
	:param sampling_type:
	:param sampling:
	:return:
	"""
	# the flows should be a deep copy of original flows. copy.deepcopy(flow)

	fid, times, sizes = flow
	sampling_data = []

	if sampling_type == 'rate':  # sampling_rate within flows.

		# The length in time of this small window is what we’re calling sampling rate.
		# features obtained on sampling_rate = 0.1 means that:
		#  1) split each flow into small windows, each window has 0.1 duration (the length in time of each small window)
		#  2) obtain the number of packets in each window (0.1s).
		#  3) all the number of packets in each window make up of the features.

		if sampling_feature in ['samp_num', 'samp_size']:
			features = []
			samp_sub = 0
			# print(f'len(times): {len(times)}, duration: {max(times)-min(times)}, sampling: {sampling},
			# num_features: {int(np.round((max(times)-min(times))/sampling))}')
			for i in range(len(times)):  # times: the arrival time of each packet
				if i == 0:
					current = times[0]
					if sampling_feature == 'samp_num':
						samp_sub = 1
					elif sampling_feature == 'samp_size':
						samp_sub = sizes[0]
					continue
				if times[i] - current <= sampling:  # interval
					if sampling_feature == 'samp_num':
						samp_sub += 1
					elif sampling_feature == 'samp_size':
						samp_sub += sizes[i]
					else:
						print(f'{sampling_feature} is not implemented yet')
				else:  # if times[i]-current > sampling:    # interval
					current += sampling
					features.append(samp_sub)
					# the time diff between times[i] and times[i-1] will be larger than mutli-samplings
					# for example, times[i]=10.0s, times[i-1]=2.0s, sampling=0.1,
					# for this case, we should insert int((10.0-2.0)//0.1) * [0]
					num_intervals = int(np.floor((times[i] - current) // sampling))
					if num_intervals > 0:
						num_intervals = min(num_intervals, 500)
						features.extend([0] * num_intervals)
						current += num_intervals * sampling
					# if current + sampling <= times[i]:  # move current to the nearest position to time[i]
					#     feat_lst_tmp, current = handle_large_time_diff(start_time=current, end_time=times[i],
					#                                                    interval=sampling)
					# features.extend(feat_lst_tmp)
					if len(features) > 500:  # avoid num_features too large to excess the memory.
						return fid, features[:500]

					# samp_sub = 1  # includes the time[i] as a new time interval
					if sampling_feature == 'samp_num':
						samp_sub = 1
					elif sampling_feature == 'samp_size':
						samp_sub = sizes[i]

			if samp_sub > 0:  # handle the last sub period in the flow.
				features.append(samp_sub)

			return fid, features
		else:
			raise ValueError(f'sampling_feature: {sampling_feature} is not implemented.')
	else:
		raise ValueError(f'sample_type: {sampling_type} is not implemented.')


def _flows_to_samps(flows, sampling_type='rate', sampling=None,
                    sampling_feature='samp_num',
                    verbose=True):
	""" sampling packets in flows
	Parameters
	----------
	flows
	sampling_type
	sampling
	sampling_feature
	verbose

	Returns
	-------

	"""
	# the flows should be a deep copy of original flows. copy.deepcopy(flows)
	# flows = copy.deepcopy(flows_lst)

	# samp_flows = []
	features = []
	features_header = []
	for fid, times, pkts in flows:
		sizes = [len(pkt) for pkt in pkts]
		if sampling_feature == 'samp_num_size':
			samp_features = []
			samp_fid_1, samp_features_1 = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
			                                               sampling=sampling, sampling_feature='samp_num')

			samp_fid_2, samp_features_2 = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
			                                               sampling=sampling, sampling_feature='samp_size')
			for i in range(len(samp_features_1)):
				if len(samp_features) > 500:
					break
				samp_features.extend([samp_features_1[i], samp_features_2[i]])
			samp_fid = samp_fid_1
		else:
			samp_fid, samp_features = sampling_packets((fid, times, sizes), sampling_type=sampling_type,
			                                           sampling=sampling, sampling_feature=sampling_feature)

		features.append((samp_fid, samp_features))  # (fid, np.array())

	# if header:
	#     head_len = int(np.quantile([len(head) for (fid, head) in features_header], q=q_iat))
	#     for i, (fid_head, fid_feat) in enumerate(list(zip(features_header, features))):
	#         fid, head = fid_head
	#         fid, feat = fid_feat
	#         if len(head) > head_len:
	#             head = head[:head_len]
	#         else:
	#             head += [0] * (head_len - len(head))
	#         features[i] = (fid, np.asarray(head + list(feat)))

	if verbose:  # for debug
		show_len = 10  # only show the first 20 difference
		samp_lens = np.asarray([len(samp_features) for (fid, samp_features) in features])[:show_len]
		raw_lens = np.asarray([max(times) - min(times) for (fid, times, sizes) in flows])[:show_len]
		print(f'(flow duration, num_windows), when sampling_rate({sampling})):\n{list(zip(raw_lens, samp_lens))}')

	return features


def _get_statistical_info(data):
	"""

	Parameters
	----------
	data: len(pkt)

	Returns
	-------
		a list includes mean, median, std, q1, q2, q3, min, and max.

	"""
	q1, q2, q3 = np.quantile(data, q=[0.25, 0.5, 0.75])  # q should be [0,1] and q2 is np.median(data)
	return [np.mean(data), np.std(data), q1, q2, q3, np.min(data), np.max(data)]


def _flows_to_stats(flows):
	'''Converts flows to FFT features

	   Arguments:
		 flows (list) = representation returned from read_pcap


	   Returns:
		 features (list) = [(fid, (max, min, ... ))]
	'''
	# the flows should be a deep copy of original flows. copy.deepcopy(flows)
	# flows = copy.deepcopy(flows_lst)

	# convert Unix timestamp arrival times into interpacket intervals
	flows = [(fid, np.diff(times), pkts) for (fid, times, pkts) in flows]  # No need to use sizes[1:]
	# len(np.diff(times)) + 1  == len(sizes)
	features = []
	features_header = []
	for fid, times, pkts in flows:  # fid, IAT, pkt_len
		sizes = [len(pkt) for pkt in pkts]
		sub_duration = sum(times)  # Note: times here actually is the results of np.diff()
		num_pkts = len(sizes)  # number of packets in the flow
		num_bytes = sum(sizes)  # all bytes in sub_duration  sum(len(pkt))
		if sub_duration == 0:
			pkts_rate = 0.0
			bytes_rate = 0.0
		else:
			pkts_rate = num_pkts / sub_duration  # it will be very larger due to the very small sub_duration
			bytes_rate = num_bytes / sub_duration
		base_feature = [sub_duration, pkts_rate, bytes_rate] + _get_statistical_info(sizes)

		features.append(
			(fid, np.asarray([np.float64(v) for v in base_feature], dtype=np.float64)))  # (fid, np.array())

	return features
