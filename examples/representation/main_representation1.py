""" Main entry point:

	Instructions:
		# Note: run the below command under the root directory of the project.
		PYTHONPATH=. python3.7 examples/representation/main_representation1.py

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
from examples.representation._constants import *  # should in the top.
import itertools

import configargparse
import pandas as pd
from joblib import Parallel, delayed

from examples.representation import _representation
from odet.utils.tool import dump, timer, check_path, remove_file

RESULT_DIR = f'results/{START_TIME}'
OUT_DIR = r'examples/representation/out'
demo = True
if demo:
	DATASETS = ['UCHI(SCAM_2019)']
	# FEATURES = ['IAT']
	# # FEATURES += ['FFT_' + v for v in FEATURES]
	# HEADER = [False]
	# MODELS = ['PCA']
	# TUNING = [False]

	# DATASETS = [
	# 	'UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)',
	# 	'UNB(PC5)',
	# 	'MAWI',
	# 	'CTU',
	# 	'UCHI(SFRIG_2021)',
	# 	'UCHI(SMTV_2019)', 'UCHI(GHOME_2019)', 'UCHI(SCAM_2019)', 'UCHI(BSTCH_2019)'
	# ]
	FEATURES = [
		'IAT',
		'SIZE', 'IAT+SIZE',
		'STATS',
		'SAMP_NUM',
		'SAMP_SIZE'
	]
	FEATURES += ['FFT_' + v for v in FEATURES]
	MODELS = ['OCSVM', 'GMM', 'PCA', 'KDE', 'IF']
	HEADER = [False, True]
	TUNING = [False, True]
else:
	pass

msg = f'HEADER: {HEADER}\n' \
      f'TUNING: {TUNING}\n' \
      f'FEATURES:{FEATURES}\n' \
      f'DATASETS: {DATASETS}\n' \
      f'MODELS: {MODELS}\n'
lg.info(msg)


def parser():
	""" parser commands

	Returns
	-------
		args: class (Namespace)
	"""
	p = configargparse.ArgParser()
	p.add_argument('-m', '--model', default='OCSVM', type=str, required=False, help='model name')
	p.add_argument('-p', '--model_params', default={'q': 0.3}, type=str, required=False, help='model params')
	p.add_argument('-t', '--tuning', default=False, type=str, required=False, help='tuning params')
	p.add_argument('-d', '--dataset', default='UCHI(SFRIG_2021)', type=str, help='dataset')
	p.add_argument('-D', '--direction', default='src', type=str, help='flow direction (src or src+dst)')
	p.add_argument('-H', '--header', default=True, type=bool, help='header')
	p.add_argument('-f', '--feature', default='FFT_IAT', type=str, help='IAT+SIZE')
	p.add_argument('-v', '--verbose', default=10, type=int, help='verbose')
	p.add_argument('-W', '--overwrite', default=False, type=bool, help='overwrite')
	p.add_argument('-o', '--out_dir', default='examples/representation/out', type=str, help='output directory')

	args = p.parse_args()
	return args


def save2txt(data, out_file, delimiter=','):
	""" Save result to txt

	Parameters
	----------
	data: list

	out_file: path
	delimiter: ','

	Returns
	-------

	"""
	with open(out_file, 'w') as f:
		for vs in data:
			line = ''
			for i, v in enumerate(vs):
				if type(v) == dict:
					score = v['score']
					line += str(score) + delimiter
					data = v['model']['data']
					line += '|'.join([str(data[0].shape[0]), str(data[2].shape[0]), str(data[4].shape[0])]) + delimiter
					line += str(data[0].shape[1]) + delimiter
					model_params = v['model']['model']['model_params']  # filter GMM
					model_params = ' '.join([f'{_k}:{str(_v)[:20]}' for _k, _v in model_params.items()])
					line += model_params.replace('\n', '') + delimiter
				else:
					line += str(v).replace('\n', '') + delimiter
			f.write(line + '\n')


def clean(in_dir=''):
	""" clean data

	Parameters
	----------
	in_dir

	Returns
	-------

	"""
	lg.debug(f'before removing the files, please double check the directory and path: {in_dir}')
	exit(-1)
	for dataset, feature, header, model, tuning in list(itertools.product(DATASETS,
	                                                                      FEATURES, HEADER, MODELS, TUNING)):
		f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}', 'Xy')
		os.remove(f + '.dat')
		os.remove(f + '.csv')


@timer
def _main():
	""" Main function

	Returns
	-------

	"""
	res = []
	out_file = f'{OUT_DIR}/src/{RESULT_DIR}/res.dat'
	is_parallel = False
	if is_parallel:  # with parallel
		def set_args(dataset, feature, header, model, tuning):
			args = parser()
			args.dataset = dataset
			args.feature = feature
			args.header = header
			args.model = model
			args.tuning = tuning
			lg.debug(args)
			return args

		# if backend='loky', the time taken is less than that of serial. but if backend='multiprocessing', we can
		# get very similar time cost comparing with serial.
		_res = []
		with Parallel(n_jobs=20, backend='loky') as parallel:
			_res = parallel(delayed(_representation.main_no_tuning_vs_tuning)  # delayed
			                (set_args(dataset, feature, header, model, tuning))  # params
			                for dataset, feature, header, model, tuning in
			                list(itertools.product(DATASETS, FEATURES, HEADER, MODELS, TUNING))  # for
			                )  # parallel
		# reorganize results
		res = []
		for history, (dataset, feature, header, model, tuning) in zip(_res, list(
				itertools.product(DATASETS, FEATURES, HEADER, MODELS, TUNING))):
			res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history['best']])
	else:  # without parallel
		for dataset, feature, header, model, tuning in list(itertools.product(DATASETS,
		                                                                      FEATURES, HEADER, MODELS, TUNING)):
			try:
				lg.info(f'*** {dataset}-{feature}-header_{header}, {model}-tuning_{tuning}')
				args = parser()
				args.dataset = dataset
				args.feature = feature
				args.header = header
				args.model = model
				args.tuning = tuning
				args.overwrite = OVERWRITE
				history = _representation.main_no_tuning_vs_tuning(args)
				res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history['best']])
				# avoid losing any result, so save it immediately.
				_out_file = f'{args.out_dir}/{args.direction}/{RESULT_DIR}/~res.csv'
				check_path(_out_file)
				save2txt(res, _out_file, delimiter=',')
			except Exception as e:
				lg.error(f'Error: {e}. [{dataset}, {feature}, {header}, {model}, {tuning}]')

	# save the final results: '.dat' and '.csv'
	check_path(out_file)
	dump(res, out_file)
	out_file = os.path.splitext(out_file)[0] + '.csv'
	remove_file(out_file, OVERWRITE)
	save2txt(res, out_file, delimiter=',')
	lg.info(f'final result: {out_file}')


@timer
def gather(in_dir='src', out_dir=''):
	""" collect all individual results together

	Parameters
	----------
	in_dir:
		search results from the given directory
	out_dir:
		save the gathered results to the given directory
	Returns
	-------
	out_file:
		the short csv for a quick overview
	"""
	res = []
	for dataset, feature, header, model, tuning in list(itertools.product(DATASETS,
	                                                                      FEATURES, HEADER, MODELS, TUNING)):
		f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}', 'res.csv')
		try:
			line = [str(v) for v in pd.read_csv(f, sep=',', header=None).values.flatten().tolist()][1:]
			lg.debug(f, line)
			if len(str(line[0])) == 0:
				lg.error(f'Error: {line}. [{header}, {tuning}, {feature}, {dataset}, {model}]')
		except Exception as e:
			lg.error(f'Error: {e}. [{header}, {tuning}, {feature}, {dataset}, {model}]')
			line = ['', '0_0|0_0|0_0', '']  # [score, shape, params]
		res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}'] + line)

	# Save all results to gather.csv
	out_file = os.path.join(out_dir, 'gather.csv')
	check_path(out_file)
	with open(out_file, 'w') as f:
		for vs in res:
			f.write(','.join(vs) + '\n')

	# Only save needed data for quick overview
	short_file = os.path.join(os.path.split(out_file)[0], 'short.csv')
	with open(short_file, 'w') as f:
		for vs in res:
			if vs[5] == '' or vs[7] == '':
				lg.warning(f'Warning: {vs}.')
			tmp = vs[6].split('|')
			shape = '|'.join(v.split('_')[0] for v in tmp)
			dim = tmp[0].split('_')[1]
			f.write(','.join(vs[:6] + [shape, dim]) + '\n')

	return out_file


@timer
def main():
	# # clean()

	# 1. Run the main function and get the results for the given parameters
	try:
		_main()
	except Exception as e:
		lg.error(f'Error: {e}.')

	# 2. Gather all the individual result
	try:
		in_dir = f'{OUT_DIR}/src'
		out_file = gather(in_dir, out_dir=os.path.join(in_dir, RESULT_DIR))
		lg.info(out_file)
	except Exception as e:
		lg.error(f'Error: {e}.')


if __name__ == '__main__':
	main()
