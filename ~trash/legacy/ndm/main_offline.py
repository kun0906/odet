""" Main function for the offline application

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 applications/offline/offline.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os
import sys
import traceback

import configargparse

from examples.offline._constants import *
from examples.offline import _offline
from odet.utils.tool import dump, timing

@timing
def offline_default_main(args):
	model = args.full_model  # full model name
	params = {
		'is_kjl': False, 'kjl_d': 5, 'kjl_n': 100, 'kjl_q': 0.9,
		'is_nystrom': False, 'nystrom_d': 5, 'nystrom_n': 100, 'nystrom_q': 0.9,
		'   is_qs': False, 'qs_k': 5, 'qs_q': 0.9
	}
	for k, v in params.items():
		# self.args.k = v
		setattr(args, k, v)

	if model == 'OCSVM(rbf)':
		args.model = 'OCSVM'
		args.kernel = 'rbf'
	elif model == "KJL-OCSVM(linear)":
		args.model = 'OCSVM'
		args.kernel = 'linear'
		args.is_kjl = True
	elif model == "Nystrom-OCSVM(linear)":
		args.model = 'OCSVM'
		args.kernel = 'linear'
		args.is_nystrom = True

	elif model in ["KJL-GMM(full)", "KJL-GMM(diag)"]:
		args.is_kjl = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		else:
			args.covariance_type = 'diag'

	elif model in ["KJL-QS-GMM(full)", "KJL-QS-GMM(diag)"]:
		args.is_kjl = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		else:
			args.covariance_type = 'diag'

	elif model in ["KJL-QS-init_GMM(full)", "KJL-QS-init_GMM(diag)"]:
		args.is_kjl = True
		args.is_qs = True  # using quickshift
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		else:
			args.covariance_type = 'diag'

	elif model in ["Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)"]:
		args.is_nystrom = True
		args.is_qs = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		else:
			args.covariance_type = 'diag'
	else:
		msg = f"{model}"
		raise NotImplementedError(msg)

	return _offline.main(args)


@timer
def offline_best_main(args):
	model = args.full_model  # full model name
	q = 0.9
	params = {'q': 0.9,
	          'is_kjl': False, 'kjl_d': 5, 'kjl_n': 100, 'kjl_q': q,
	          'is_nystrom': False, 'nystrom_d': 5, 'nystrom_n': 100, 'nystrom_q': q,
	          '   is_qs': False, 'qs_k': 5, 'qs_q': 0.3
	          }
	for k, v in params.items():
		# self.args.k = v
		setattr(args, k, v)
	qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	components = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

	res = {'auc': 0}

	if model == 'OCSVM(rbf)':
		args.model = 'OCSVM'
		args.kernel = 'rbf'
		for _q in qs:
			args.q = _q
			_res = _offline.main(args)
			if _res['auc'] > res['auc']:
				res = _res

	elif model == "KJL-OCSVM(linear)":
		args.model = 'OCSVM'
		args.kernel = 'linear'
		args.is_kjl = True
		for _q in qs:
			args.q = _q
			args.kjl_q = _q
			_res = _offline.main(args)
			if _res['auc'] > res['auc']:
				res = _res

	elif model == "Nystrom-OCSVM(linear)":
		args.model = 'OCSVM'
		args.kernel = 'linear'
		args.is_nystrom = True
		for _q in qs:
			args.q = _q
			args.nystrom_q = _q
			_res = _offline.main(args)
			if _res['auc'] > res['auc']:
				res = _res

	elif model in ["KJL-GMM(full)", "KJL-GMM(diag)"]:
		args.is_kjl = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		elif 'diag' in model:
			args.covariance_type = 'diag'
		else:
			msg = model
			raise NotImplementedError(msg)
		for _q in qs:
			for _comps in components:
				args.q = _q
				args.kjl_q = _q
				args.comps = _comps
				_res = _offline.main(args)
				if _res['auc'] > res['auc']:
					res = _res

	elif model in ["KJL-QS-GMM(full)", "KJL-QS-GMM(diag)"]:
		args.is_kjl = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		elif 'diag' in model:
			args.covariance_type = 'diag'
		else:
			msg = model
			raise NotImplementedError(msg)

		args.is_qs = True
		for _q in qs:
			for _comps in components:
				args.q = _q
				args.kjl_q = _q
				args.comps = _comps
				_res = _offline.main(args)
				if _res['auc'] > res['auc']:
					res = _res

	elif model in ["KJL-QS-init_GMM(full)", "KJL-QS-init_GMM(diag)"]:
		args.is_kjl = True
		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		elif 'diag' in model:
			args.covariance_type = 'diag'
		else:
			msg = model
			raise NotImplementedError(msg)

		args.is_qs = True
		args.is_qs_init = True
		for _q in qs:
			for _comps in components:
				args.q = _q
				args.kjl_q = _q
				args.comps = _comps
				_res = _offline.main(args)
				if _res['auc'] > res['auc']:
					res = _res

	elif model in ["Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)"]:
		args.is_nystrom = True
		args.is_qs = True
		args.model = 'GMM'
		args.covariance_type = 'diag'

		args.model = 'GMM'
		if 'full' in model:
			args.covariance_type = 'full'
		elif 'diag' in model:
			args.covariance_type = 'diag'
		else:
			msg = model
			raise NotImplementedError(msg)

		args.is_qs = True
		args.is_qs_init = True
		for _q in qs:
			for _comps in components:
				args.q = _q
				args.kjl_q = _q
				args.comps = _comps
				_res = _offline.main(args)
				if _res['auc'] > res['auc']:
					res = _res

	else:
		msg = f"{model}"
		raise NotImplementedError(msg)

	return res


class Args:
	def __init__(self, dataset, model, overwrite=False, out_dir='applications/offline/out',
	             verbose=10, random_state=42):
		p = configargparse.ArgParser()
		p.add_argument('-m', '--full_model', default='OCSVM', type=str, required=False, help='full model name')
		p.add_argument('-d', '--dataset', default='UNB', type=str, help='dataset')
		p.add_argument('-v', '--verbose', default=10, type=int, help='verbose')
		p.add_argument('-or', '--overwrite', default=False, type=bool, help='overwrite')
		p.add_argument('-o', '--out_dir', default='applications/offline/out', type=str, help='output directory')

		self.args = p.parse_args()
		self.args.dataset = dataset
		self.args.full_model = model
		self.args.verbose = verbose
		self.args.overwrite = overwrite
		self.args.out_dir = out_dir
		self.args.random_state = random_state

@timer
def main(is_defaut_parms=True):
	res = {}
	tot = len(DATASETS.keys()) * len(MODELS.keys())
	i = 1
	for dataset in DATASETS.keys():
		dataset_res = {}
		for model in MODELS.keys():
			try:
				lg.info(f'\n\n***{i}/{tot}:{dataset}_{FEATURE}-{model}-default_params_{is_defaut_parms}')
				if is_defaut_parms:
					args = Args(dataset, model)
					_res = offline_default_main(args.args)
				else:
					args = Args(dataset, model)
					_res = offline_best_main(args.args)
				dataset_res[model] = _res
			except Exception as e:
				msg = f'{dataset}-{model}-default_{is_defaut_parms}: {e}'
				lg.error(msg)
				traceback.print_exc()
			i+=1
		res[dataset] = dataset_res

	out_file = os.path.join(OUT_DIR, f'{FEATURE}-default_{is_defaut_parms}.dat')
	dump(res, out_file)

	return res


if __name__ == '__main__':
	main()
