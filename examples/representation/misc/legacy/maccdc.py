""" Base class for dataset

"""
import os

import numpy as np
from loguru import logger as lg
from odet.pparser.parser import _get_flow_duration, _get_split_interval

from odet.datasets._base import Base
from odet.utils.tool import load, check_path, dump


class MACCDC(Base):

	def __init__(self, pcap_file='applications/offline/data/deeplens_open_shut_fridge_batch_8.pcap_filtered.pcap',
	             out_dir='.', feature_name='IAT+SIZE', overwrite=False):
		self.pcap_file = pcap_file
		self.X = None
		self.y = None
		self.overwrite = overwrite
		self.out_dir = out_dir
		self.feature_name = feature_name

		self.Xy_file = os.path.join(self.out_dir, f'UNB-{self.feature_name}-Xy.dat')
		lg.info(f'{self.Xy_file}')

	def generate(self):
		if os.path.exists(self.Xy_file):
			self.X, self.y = load(self.Xy_file)
		else:
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

		return self.X, self.y
