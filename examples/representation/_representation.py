""" Main function will calls this script

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import numpy as np

from examples.representation._constants import *
import copy
import time
import traceback
from collections import Counter

from scipy.spatial import distance
from sklearn import metrics
from sklearn.metrics import roc_curve

from examples.representation.datasets.ctu import CTU
from examples.representation.datasets.mawi import MAWI
from examples.representation.datasets.uchi import UCHI
from examples.representation.datasets.unb import UNB
from odet.ndm.ae import AE
from odet.ndm.gmm import GMM
from odet.ndm.iforest import IF
from odet.ndm.kde import KDE
from odet.ndm.ocsvm import OCSVM
from odet.ndm.pca import PCA
from odet.utils.tool import timer, check_path, dump, split_train_val_test, normalize, remove_file, check_arr


def obtain_means_init_quickshift_pp(X, k=None, beta=0.9, thres_n=20):
	from QuickshiftPP import QuickshiftPP
	"""Initialize GMM
			1) Download quickshift++ from github: https://github.com/google/quickshift/archive/refs/heads/master.zip
			2) unzip and move the folder to your project
			3) python3 setup.py build
			4) python3 setup.py install
			5) from QuickshiftPP import QuickshiftPP
		:param X_train:
		:param k:
			# k: number of neighbors in k-NN
			# beta: fluctuation parameter which ranges between 0 and 1.

		:return:
		"""
	start = time.time()
	if k <= 0 or k > X.shape[0]:
		lg.debug(f'k {k} is not correct, so change it to X.shape[0]')
		k = X.shape[0]
	lg.debug(f"number of neighbors in k-NN: {k}")
	# Declare a Quickshift++ model with tuning hyperparameters.
	model = QuickshiftPP(k=k, beta=beta)

	# Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
	ret_code = 1
	try:
		ret_code = model.fit(X)
	except Exception as e:
		msg = f'quickshift++ fit error: {e}, ret_code: {ret_code}'
		raise ValueError(msg)

	# if ret_code < 0:
	#     print(f'ret_code ({ret_code}) < 0, fit fail')
	#     raise ValueError('ret_code < 0, fit fail')

	end = time.time()
	quick_training_time = end - start

	start = time.time()
	all_labels_ = model.memberships
	all_n_clusters = len(set(all_labels_))
	cluster_centers = []
	for i in range(all_n_clusters):
		idxs = np.where(all_labels_ == i)[0]  # get index of each cluster. np.where return tuple
		if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
			continue
		center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
		cluster_centers.append(center_cluster_i)

	means_init = np.asarray(cluster_centers, dtype=float)
	n_clusters = means_init.shape[0]
	end = time.time()
	ignore_clusters_time = end - start
	lg.debug(f'*** quick_training_time: {quick_training_time}, ignore_clusters_time: {ignore_clusters_time}')
	# lg.dueg(f'--all clusters ({all_n_clusters}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
	#       f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(all_labels_)}, *** '
	#       f'len(Counter(labels_)): {all_n_clusters}')

	return means_init, n_clusters


@timer
class Data:
	def __init__(self, in_dir='../Datasets', dataset_name=None, direction='src', verbose=10, overwrite=False,
	             feature_name='IAT+SIZE', header=False,
	             out_dir='examples/representation/out', random_state=42):
		self.dataset_name = dataset_name
		self.direction = direction
		self.verbose = verbose
		self.overwrite = overwrite
		self.feature_name = feature_name
		self.header = header
		self.random_state = random_state

		if dataset_name in ['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)']:
			self.data = UNB(in_dir=in_dir, dataset_name=dataset_name, direction=direction,
			                feature_name=feature_name, header=header,
			                out_dir=out_dir,
			                overwrite=overwrite, random_state=random_state)
		elif dataset_name in ['CTU']:
			self.data = CTU(dataset_name=dataset_name, direction=direction,
			                feature_name=feature_name, header=header,
			                out_dir=out_dir,
			                overwrite=overwrite, random_state=random_state)
		elif dataset_name in ['MAWI']:
			self.data = MAWI(dataset_name=dataset_name, direction=direction,
			                 feature_name=feature_name, header=header,
			                 out_dir=out_dir,
			                 overwrite=overwrite, random_state=random_state)
		elif dataset_name in ['UCHI(SFRIG_2021)',
		                      'UCHI(GHOME_2019)', 'UCHI(SCAM_2019)', 'UCHI(BSTCH_2019)',
		                      'UCHI(SMTV_2019)']:
			self.data = UCHI(dataset_name=dataset_name, direction=direction,
			                 feature_name=feature_name, header=header,
			                 out_dir=out_dir,
			                 overwrite=overwrite, random_state=random_state)
		else:
			msg = dataset_name
			raise NotImplementedError(msg)

		self.X = None
		self.y = None

	def generate(self):
		meta = self.data.generate()
		self.X, self.y = meta['X'], meta['y']


@timer
class Model:

	def __init__(self, name='OCSVM', score_metric='auc', overwrite=OVERWRITE, random_state=RANDOM_STATE, **kwargs):
		self.name = name
		self.overwrite = overwrite
		self.random_state = random_state
		self.score_metric = score_metric
		self.model_params = kwargs['model_params']
		self.history = {'model_name': name}
		self.ss = kwargs['ss']
		self.tuning = kwargs['tuning']

	def train(self, X, y=None):
		if self.name == 'OCSVM':
			q = self.model_params['q']
			# find the best parameters of the detector
			distances = distance.pdist(X, metric='euclidean')
			sigma = np.quantile(distances, q=q)
			if sigma == 0:  # find a new non-zero sigma
				lg.debug(f'sigma: {sigma}, q: {q}')
				q_lst = list(np.linspace(q + 0.01, 1, 10, endpoint=False))
				sigma_lst = np.quantile(distances, q=q_lst)
				sigma, q = [(s_v, q_v) for (s_v, q_v) in zip(sigma_lst, q_lst) if s_v > 0][0]
			gamma = 1 / (2 * sigma ** 2)
			lg.debug(f'q: {q}, sigma: {sigma}, gamma: {gamma}, {np.quantile(distances, q=[0, 0.3, 0.9, 1])}')
			self.model = OCSVM(kernel='rbf', gamma=gamma, random_state=self.random_state)
		elif self.name == 'GMM':
			if not self.tuning or self.model_params['n_components'] == 'quickshift':
				try:
					means_init, n_components = obtain_means_init_quickshift_pp(X, k=int(np.sqrt(len(y))))
				except:
					raise ValueError('obtain_means_init_quickshift error.')
				means_init = means_init * self.ss.scale_ + self.ss.mean_
				self.model = GMM(n_components=n_components, means_init=means_init, covariance_type='diag')
			else:
				n_components = self.model_params['n_components']
				self.model = GMM(n_components=n_components, covariance_type='diag')
		elif self.name == 'AE':
			hidden_neurons = self.model_params['hidden_neurons']
			self.model = AE(hidden_neurons=hidden_neurons, random_state=self.random_state)
		elif self.name == 'PCA':
			n_components = self.model_params['n_components']
			self.model = PCA(n_components=n_components, random_state=self.random_state)
		elif self.name == 'KDE':
			q = self.model_params['q']
			# find the best parameters of the detector
			distances = distance.pdist(X, metric='euclidean')
			sigma = np.quantile(distances, q=q)
			if sigma == 0:  # find a new non-zero sigma
				lg.debug(f'sigma: {sigma}, q: {q}')
				q_lst = list(np.linspace(q + 0.01, 1, 10, endpoint=False))
				sigma_lst = np.quantile(distances, q=q_lst)
				sigma, q = [(s_v, q_v) for (s_v, q_v) in zip(sigma_lst, q_lst) if s_v > 0][0]
			lg.debug(f'q: {q}, sigma: {sigma}, {np.quantile(distances, q=[0, 0.3, 0.9, 1])}')
			self.model = KDE(bandwidth=sigma, random_state=self.random_state)
		elif self.name == 'IF':
			n_estimators = self.model_params['n_estimators']
			self.model = IF(n_estimators=n_estimators, random_state=self.random_state)
		else:
			msg = self.name
			raise NotImplementedError(msg)

		self.model.fit(X)
		self.history['model_params'] = self.model.get_params()
		self.history['model'] = copy.deepcopy(self.model)

	def test(self, X, y):
		try:
			y_score = self.model.decision_function(X)
			y_score = check_arr(y_score)  # fill Nan or inf to 0
			if self.score_metric == 'auc':
				# For binary  y_true, y_score is supposed to be the score of the class with greater label.
				# pos_label = 1, so y_score should be the corresponding score (i.e., novel score)
				fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
				self.score = metrics.auc(fpr, tpr)
			else:
				msg = f'{self.score_metric}'
				raise NotImplementedError(msg)
		except Exception as e:
			lg.error(f'Error: {e}')
			self.score = 0
		self.history['score'] = self.score
		return self.history


def save2txt(res, out_file, delimiter=','):
	""" Save results to txt

	Parameters
	----------
	res
	out_file
	delimiter

	Returns
	-------

	"""
	with open(out_file, 'w') as f:
		line = out_file + delimiter
		score = res['score']
		line += str(score) + delimiter
		data = res['data']
		line += '|'.join(['_'.join(str(v) for v in list(data[0].shape)),
		                  '_'.join(str(v) for v in list(data[2].shape)),
		                  '_'.join(str(v) for v in list(data[4].shape))]) + delimiter
		model = res['model']
		line += str(model).replace('\n', '') + delimiter
		f.write(line + '\n')


@timer
def main_no_tuning_vs_tuning(args=None):
	""" get results with default and best parameters according to the args.

	Parameters
	----------
	args: given parameters

	Returns
	-------
	history: dict
		store all the results in a dictionary
	"""
	# 1. Get dimension of the dataset. For some algorithms, they need the dimensions (e.g., AE)
	data = Data(dataset_name=args.dataset, direction=args.direction, feature_name=args.feature, header=args.header,
	            overwrite=args.overwrite, random_state=RANDOM_STATE)
	data.generate()
	if 'SAMP' in args.feature:
		X = data.X[0]
	else:
		X = data.X

	# 2. Get the results with the given model
	if args.model == 'OCSVM':
		if args.tuning:
			qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
		else:
			qs = [0.3]
		history = {}  # store the best result, model parameters, and the best model (dict)
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: q = {qs}')
		for q in qs:
			args.model_params = {'q': q}
			# get results on the validation set
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['q'] = q
				best['model'] = copy.deepcopy(history_)
			history[q] = history_

		# get the final result on the test set.
		args.model_params = {'q': best['q']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best
	elif args.model == 'GMM':
		if args.tuning:
			n_components_arr = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
		else:
			n_components_arr = ['quickshift']
		history = {}
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: q = {n_components_arr}')
		for n_components in n_components_arr:
			args.model_params = {'n_components': n_components}
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['n_components'] = n_components
				best['model'] = copy.deepcopy(history_)
			history[n_components] = history_

		# get the final result on the test set.
		args.model_params = {'n_components': best['n_components']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best

	elif args.model == 'IF':
		if args.tuning:
			n_estimators_arr = [int(v) for v in list(np.linspace(30, 300, num=10, endpoint=True))]
		else:
			n_estimators_arr = [100]
		history = {}
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: n_estimators_arr = {n_estimators_arr}')
		for n_estimators in n_estimators_arr:
			args.model_params = {'n_estimators': n_estimators}
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['n_estimators'] = n_estimators
				best['model'] = copy.deepcopy(history_)
			history[n_estimators] = history_

		# get the final result on the test set.
		args.model_params = {'n_estimators': best['n_estimators']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best

	elif args.model == 'PCA':
		if args.tuning:
			n_components_arr = [int(v) for v in list(np.linspace(1, min(X.shape), num=10, endpoint=False))]
		else:
			n_components_arr = ['mle']
		history = {}
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: n_components_arr = {n_components_arr}')
		for n_components in n_components_arr:
			args.model_params = {'n_components': n_components}
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['n_components'] = n_components
				best['model'] = copy.deepcopy(history_)
			history[n_components] = history_

		# get the final result on the test set.
		args.model_params = {'n_components': best['n_components']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best
	elif args.model == 'KDE':
		if args.tuning:
			qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
		else:
			qs = [0.3]
		history = {}
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: q = {qs}')
		for q in qs:
			args.model_params = {'q': q}
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['q'] = q
				best['model'] = copy.deepcopy(history_)
			history[q] = history_
		# get the final result on the test set.
		args.model_params = {'q': best['q']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best

	elif args.model == 'AE':
		if args.tuning:
			feat_dim = X.shape[1]

			def get_AE_parameters(d, num=10):
				latent_sizes = []
				for i in range(num):
					v = np.ceil(1 + i * (d - 2) / 9).astype(int)
					if v not in latent_sizes:
						latent_sizes.append(v)

				hidden_sizes = [min((d - 1), np.ceil(2 * v).astype(int)) for v in latent_sizes]

				hidden_neurons = []
				for i, (hid, lat) in enumerate(zip(hidden_sizes, latent_sizes)):
					v = [d, hid, lat, hid, d]
					hidden_neurons.append(v)
				return hidden_neurons

			hidden_neurons_arr = get_AE_parameters(feat_dim, num=10)
		else:
			feat_dim = X.shape[1]
			latent_dim = np.ceil(feat_dim / 2).astype(int)
			hid = min((feat_dim - 1), np.ceil(2 * latent_dim).astype(int))
			hidden_neurons = [feat_dim, hid, latent_dim, hid, feat_dim]
			hidden_neurons_arr = [hidden_neurons]

		history = {}
		best = {'score': 0, 'model': None}
		lg.debug(f'Tuning: hidden_neurons = {hidden_neurons_arr}')
		for hidden_neurons in hidden_neurons_arr:
			args.model_params = {'hidden_neurons': hidden_neurons}
			history_ = main(args, test=False)
			score_ = history_['score']
			if score_ > best['score']:
				best['score'] = score_
				best['hidden_neurons'] = hidden_neurons
				best['model'] = copy.deepcopy(history_)
			history[tuple(hidden_neurons)] = history_
		# get the final result on the test set.
		args.model_params = {'hidden_neurons': best['hidden_neurons']}
		best['model'] = main(args, test=True)
		best['score'] = best['model']['score']
		history['best'] = best

	else:
		msg = f'{args.model}'
		raise NotImplementedError(msg)
	# lg.info(f'\n*** best: ' + str(history['best']))
	out_file = os.path.join(args.out_dir, args.direction, args.dataset, args.feature, f'header_{args.header}',
	                        args.model, f'tuning_{args.tuning}', 'res.dat')
	check_path(out_file)
	dump(history, out_file)

	return history


def _single_main(args, X, y, test=False):
	""" Get the result on given parameters

	Parameters
	----------
	args
	X
	y
	test

	Returns
	-------
	res: evalated results
	data: (train, val, test)
	"""
	lg.debug(args)
	lg.debug(f'X.shape: {X.shape}, y: {Counter(y)}')
	###############################################################################################################
	""" 1.2 Split train and test set

	"""
	lg.info(f'\n--- 1.2 Split train and test set')
	X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y,
	                                                                      shuffle=True,
	                                                                      random_state=RANDOM_STATE)
	# normalization
	ss, X_train, y_train, X_val, y_val, X_test, y_test = normalize(X_train, y_train, X_val, y_val, X_test,
	                                                               y_test)
	lg.debug(f'X_train:{X_train.shape}, y_train: {Counter(y_train)}')
	lg.debug(f'X_val:{X_val.shape}, y_val: {Counter(y_val)}')
	lg.debug(f'X_test:{X_test.shape}, y_test: {Counter(y_test)}')

	###############################################################################################################
	""" 2.1 Build the model

	"""
	lg.info(f'\n--- 2.1 Build the model')
	model = Model(name=args.model, model_params=args.model_params, ss=ss, tuning=args.tuning,
	              overwrite=args.overwrite, random_state=RANDOM_STATE)
	model.train(X_train, y_train)

	###############################################################################################################
	""" 2.2 Evaluate the model

	"""
	lg.info(f'\n--- 2.2 Evaluate the model')
	if not test:
		res = model.test(X_val, y_val)
	else:
		res = model.test(X_test, y_test)
	lg.info(res)

	data = (X_train, y_train, X_val, y_val, X_test, y_test)
	return res, data


@timer
def main(args=None, test=False):
	""" Get the result according to the given parameters

	Parameters
	----------
	args
	test: boolean
		if we evaluate the built model on val set or test set
	Returns
	-------
	history: dict
		Return the best result on 'SAMP' related feature. Otherwise, return the result
	"""
	try:
		lg.debug(args)
		out_dir = os.path.join(args.out_dir, args.direction, args.dataset, args.feature, f'header_{args.header}',
		                       args.model, f'tuning_{args.tuning}')

		###############################################################################################################
		""" 1.1 Parse data and extract features
			
		"""
		lg.info(f'\n--- 1.1 Parse data')
		data = Data(dataset_name=args.dataset, direction=args.direction, feature_name=args.feature, header=args.header,
		            overwrite=args.overwrite, random_state=RANDOM_STATE)
		data.generate()

		if 'SAMP' in args.feature:
			best = {'score': 0, 'model': None}
			for i, (X, y) in enumerate(zip(data.X, data.y)):
				lg.debug(f'SAMP_{i}')
				try:
					res_, data_ = _single_main(args, X, y, test=test)
				except Exception as e:
					lg.error(f'Error: {e}. SAMP_{i}')
					continue
				# get the best results on SAMP data
				if res_['score'] > best['score']:
					best['score'] = res_['score']
					best['model'] = copy.deepcopy(res_)
					best['data'] = copy.deepcopy(data_)
			history = best
		else:
			X, y = data.X, data.y
			res_, data_ = _single_main(args, X, y, test=test)
			history = {'score': res_['score'], 'model': res_, 'data': data_}

	except Exception as e:
		traceback.print_exc()
		history = {'score': 0, 'model': {}, 'data': (None, None, None, None, None, None)}

	###############################################################################################################
	""" 3. Dump the result to disk

	"""
	lg.info(f'\n--- 3. Save the result')
	out_file = os.path.join(out_dir, f'res.dat')
	check_path(out_file)
	dump(history, out_file=out_file)
	out_file = os.path.splitext(out_file)[0] + '.csv'
	remove_file(out_file, overwrite=OVERWRITE)
	save2txt(history, out_file)
	lg.info(f'res_file: {out_file}')

	return history


if __name__ == '__main__':
	main()
