""" Includes all configurations, such as constants and global random_state.
    1. set a random seed for os, random, and so on.
    2. constants
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import datetime
import os
import random
import sys

import numpy as np

#############################################################################################
# 1. random state control in order to achieve reproductive results
# ref: https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
START_TIME = datetime.datetime.now()
print(START_TIME)
print(sys.path)
# Seed value
# Apparently you may use different seed values at each stage
RANDOM_STATE = 42
# 1). Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# 2). Set the `python` built-in pseudo-random generator at a fixed value
random.seed(RANDOM_STATE)

# 3). Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(RANDOM_STATE)
#
# # 4). set torch
# import torch
#
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#############################################################################################
"""2. Constant
"""
OVERWRITE = False
data_orig2name = {'UNB(PC1)': 'UNB(PC1)',
                  'UNB(PC2)': 'UNB(PC2)',
                  'UNB(PC3)': 'UNB(PC3)',
                  'UNB(PC4)': 'UNB(PC4)',
                  'UNB(PC5)': 'UNB(PC5)',
                  'CTU': 'CTU',
                  'MAWI': 'MAWI',
                  'UCHI(SMTV_2019)': 'TV&RT',
                  'UCHI(SFRIG_2021)': 'SFrig',
                  'UCHI(GHOME_2019)': 'GHom',
                  'UCHI(SCAM_2019)': 'SCam',
                  'UCHI(BSTCH_2019)': 'BSTch',
                  }
#
# data_orig2name = {
#     'UNB/CICIDS_2017/pc_192.168.10.5': 'UNB(PC1)',
#     'UNB/CICIDS_2017/pc_192.168.10.8':'UNB(PC2)',
#     'UNB/CICIDS_2017/pc_192.168.10.9':'UNB(PC3)',
#     'UNB/CICIDS_2017/pc_192.168.10.14':'UNB(PC4)',
#     'UNB/CICIDS_2017/pc_192.168.10.15':'UNB(PC5)',
#     'CTU/IOT_2017/pc_10.0.2.15': 'CTU',
#
#     'MAWI/WIDE_2019/pc_202.171.168.50': 'MAWI',
#     # 'MAWI/WIDE_2020/pc_203.78.7.165',
#     # # 'MAWI/WIDE_2020/pc_202.75.33.114',
#     # 'MAWI/WIDE_2020/pc_23.222.78.164',
#     # 'MAWI/WIDE_2020/pc_203.78.4.32',
#     # 'MAWI/WIDE_2020/pc_203.78.8.151',
#     # 'MAWI/WIDE_2020/pc_203.78.4.32',
#     # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
#     # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
#
#     'UCHI/IOT_2019/smtv_10.42.0.1':'TV&RT',
#
#     'UCHI/IOT_2019/ghome_192.168.143.20':'GHom',
#     'UCHI/IOT_2019/scam_192.168.143.42':'SCam',
#     'UCHI/IOT_2019/sfrig_192.168.143.43':'Sfrig',
#     'UCHI/IOT_2019/bstch_192.168.143.48':'BSTch'
# }

data_name2orig = {v: k for k, v in data_orig2name.items()}

# might not needed
model_orig2name = {}

model_name2orig = {v: k for k, v in model_orig2name.items()}

DATASETS = [
		'UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)',
		'UNB(PC5)',
		'MAWI',
		'CTU',
		'UCHI(SFRIG_2021)',
		'UCHI(SMTV_2019)', 'UCHI(GHOME_2019)', 'UCHI(SCAM_2019)', 'UCHI(BSTCH_2019)'
	]
FEATURES = [
    'IAT',
    'SIZE', 'IAT+SIZE',
    'STATS',
    'SAMP_NUM',
    'SAMP_SIZE'
]
FEATURES += ['FFT_' + v for v in FEATURES]
HEADER = [False, True]
MODELS = ['OCSVM', 'GMM', 'AE', 'PCA', 'KDE', 'IF']
TUNING = [False, True]

#############################################################################################
"""3. log 

"""
from loguru import logger as lg

lg.remove()
lg.add(sys.stdout, level='DEBUG')


