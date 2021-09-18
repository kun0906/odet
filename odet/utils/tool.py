"""Common functions.

"""
# Authors: kun.bj@outlook.com
#
# License: xxx

import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import assert_all_finite


def dump(data, out_file=''):
	"""Save data to file

	Parameters
	----------
	data: any data

	out_file: str
		out file path

	Returns
	-------

	"""
	if os.path.exists(out_file):
		os.remove(out_file)
	# save results
	with open(out_file, 'wb') as out_hdl:
		pickle.dump(data, out_hdl)


def load(in_file):
	"""load data from file

	Parameters
	----------
	in_file: str
		input file path

	Returns
	-------
	data:
		loaded data
	"""
	with open(in_file, 'rb') as f:
		data = pickle.load(f)

	return data


def data_info(data=None, name='data'):
	"""Print data basic information

	Parameters
	----------
	data: array

	name: str
		data name

	Returns
	-------

	"""

	pd.set_option('display.max_rows', 500)
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 100)
	pd.set_option('display.float_format', lambda x: '%.3f' % x)  # without scientific notation

	columns = ['col_' + str(i) for i in range(data.shape[1])]
	dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
	print(f'{name}.shape: {data.shape}')
	print(dataset.describe())
	print(dataset.info(verbose=True))


def check_path(file_path):
	"""Check if a path is existed or not.
	 If the path doesn't exist, then create it.

	Parameters
	----------
	file_path: str


	Returns
	-------

	"""
	path_dir = os.path.dirname(file_path)
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)

	return file_path


def timer(func):
	# This function shows the execution time of
	# the function object passed
	def wrap_func(*args, **kwargs):
		t1 = time.time()
		result = func(*args, **kwargs)
		t2 = time.time()
		print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
		return result

	return wrap_func



def time_func(func, *args, **kwargs):
	start = datetime.now()

	result = func(*args, **kwargs)

	end = datetime.now()
	total_time = (end - start).total_seconds()
	# print(f'{func} running time: {total_time}, and result: {result}')

	return result, total_time


def get_file_path(**kwargs):
	"""

	Parameters
	----------
	kwargs

	Returns
	-------

	"""
	path_dir = ''
	file_name = ''
	for (k, v) in kwargs.items():
		if k == 'file_name':
			file_name = v
			continue
		path_dir = os.path.join(path_dir, v)

	return os.path.join(path_dir, file_name)


def split_train_val_test(X, y, shuffle=True, random_state=42):
	"""

	Parameters
	----------
	X
	y
	shuffle
	random_state

	Returns
	-------

	"""
	# split normal and abnormal
	X_normal = X[y == 0]
	y_normal = y[y == 0]
	X_abnormal = X[y == 1]
	y_abnormal = y[y == 1]
	# if len(y_normal) > 20000:
	# 	X_normal, _, y_normal, _ = train_test_split(X_normal, y_normal, train_size= 20000,
	# 	                                                stratify=y_normal, random_state=random_state)
	if len(y_abnormal) > 4000:
		X_abnormal, _, y_abnormal, _ = train_test_split(X_abnormal, y_abnormal, train_size=4000,
		                                                stratify=y_abnormal, random_state=random_state)

	# get test set first
	X_normal, X_test, y_normal, y_test = train_test_split(X_normal, y_normal, test_size=len(y_abnormal),
	                                                      shuffle=shuffle, random_state=random_state)
	X_test = np.concatenate([X_test, X_abnormal], axis=0)
	y_test = np.concatenate([y_test, y_abnormal], axis=0)

	# select a part of val_data from test_set.
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
	                                                stratify=y_test, random_state=random_state)
	X_train = X_normal[:10000]
	y_train = y_normal[:10000]
	return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X_train, y_train, X_val, y_val, X_test, y_test):
	""" Normalize data

	Parameters
	----------
	X_train
	y_train
	X_val
	y_val
	X_test
	y_test

	Returns
	-------

	"""
	ss = StandardScaler()
	ss.fit(X_train)

	X_train = ss.transform(X_train)
	X_val = ss.transform(X_val)
	X_test = ss.transform(X_test)
	return ss, X_train, y_train, X_val, y_val, X_test, y_test


def remove_file(in_file, overwrite=False):
	""" remove file

	Parameters
	----------
	in_file
	overwrite

	Returns
	-------

	"""
	if overwrite:
		if os.path.exists(in_file):
			os.remove(in_file)
		else:
			pass


def remove_dir(in_dir, overwrite=False):
	""" remove directory

	Parameters
	----------
	in_dir
	overwrite

	Returns
	-------

	"""
	if overwrite:
		if os.path.exists(in_dir):
			shutil.rmtree(in_dir)
		else:
			pass

def check_arr(X):
	""" Fill nan to 0

	Parameters
	----------
	X

	Returns
	-------
	X
	"""

	# X[np.isnan(X)] = 0
	# # X[np.isneginf(X)] = 0
	# # X[np.isinf(X)] = 0
	# X[np.isfinite(X)] = 0
	# print(X)
	# print(np.quantile(X, q=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]))
	# import pandas as pd
	# df = pd.DataFrame(X)
	# print(df.describe())
	#
	# assert_all_finite(X, allow_nan=False)
	# print(f'***{np.mean(X, axis=0)}, {np.any(np.isnan(X))}, '
	#       f'{np.any(np.isinf(X))}, {np.any(np.isneginf(X))}')
	#
	X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
	return X
