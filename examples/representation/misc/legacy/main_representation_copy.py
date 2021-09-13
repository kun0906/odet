import csv
import itertools
import os
import time
import traceback
import pandas as pd
import configargparse
import numpy as np
from joblib import Parallel, delayed
from loguru import logger as lg

from examples.representation import _representation
from odet.utils.tool import dump, timer, load, check_path

DATE = f'results/{time.time()}'
demo = True
if demo:
	DATASETS = ['UNB(PC5)',
		'MAWI',
		'CTU',]
	FEATURES = ['SAMP_NUM']
	# FEATURES += ['FFT_' + v for v in FEATURES]
	HEADER = [False, True]
	lg.info(FEATURES)
	MODELS = ['AE']
	TUNING = [False, True]
else:
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
	lg.info(FEATURES)
	MODELS = ['OCSVM', 'GMM', 'AE', 'PCA', 'KDE', 'IF']
	TUNING = [False, True]


def parser():
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


def save2txt(res, out_file, delimiter=','):
	with open(out_file, 'w') as f:
		for vs in res:
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




def get_one_res(res, header, tuning, feature, dataset, model):
	# print(header, tuning, feature, dataset, model)
	for vs in res:
		dataset_, feature_, header_, model_, tuning_ = vs[0], vs[1], vs[2], vs[3], vs[4]
		if (header_ == header) and (tuning_ == tuning) and (feature_ == feature) and (dataset_ == dataset) and (
				model_ == model):
			score = vs[5]['score']
			data = vs[5]['model']['data']
			shape = '|'.join(str(v) for v in [data[0].shape[0], data[2].shape[0], data[4].shape[0]])
			dim = data[0].shape[1]
			model_params = vs[5]['model']['model']['model_params']  # filter GMM
			model_params = ' '.join([f'{_k}:{str(_v)[:20]}' for _k, _v in model_params.items()])
			return [header, tuning, feature, dataset, model, f'{score:.2f}', shape, str(dim), model_params]
	return [header, tuning, feature, dataset, model, '-1', '0|0|0', '0']

@timer
def report(in_file='gather.dat', delimiter=','):
	res = load(in_file)
	out_file = os.path.split(in_file) + 'report.csv'
	check_path(out_file)
	with open(out_file, 'w') as f:
		for header in HEADER:
			for tuning in TUNING:
				for feature in FEATURES:
					for dataset in DATASETS:
						for model in MODELS:
							data = get_one_res(res, f'header_{header}', f'tuning_{tuning}', feature, dataset, model)
							line = f'{delimiter}'.join(data) + '\n'
							lg.debug(line)
							f.write(line)

	lg.info(f'report: {out_file}')
	return out_file

def clean(in_dir = 'examples/representation/out/src'):
	print(f'before removing the files, please double check the directory and path: {in_dir}')
	exit(-1)
	for dataset in DATASETS:
		for feature in FEATURES:
			for header in HEADER:
				for model in MODELS:
					for tuning in TUNING:
						f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}', 'Xy')
						os.remove(f+'.dat')
						os.remove(f+'.csv')


@timer
def main():
	res = []
	res_file = 'res2'
	is_parallel = False
	if is_parallel:
		def set_args(dataset, feature, header, model, tuning):
			args = parser()
			args.dataset = dataset
			args.feature = feature
			args.header = header
			args.model = model
			args.tuning = tuning
			print(args)
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
		for history, (dataset, feature, header, model, tuning) in zip(_res, list(itertools.product(DATASETS,
		                                                                      FEATURES, HEADER, MODELS, TUNING))):
			res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history])
		out_file = f'examples/representation/out/src/{DATE}/{res_file}.dat'
	else:  # without parallel
		for dataset in DATASETS:
			for feature in FEATURES:
				for header in HEADER:
					for model in MODELS:
						for tuning in TUNING:
							try:
								print(f'*** {dataset}-{feature}-header_{header}, {model}-tuning_{tuning}')
								args = parser()
								args.dataset = dataset
								args.feature = feature
								args.header = header
								args.model = model
								args.tuning = tuning
								history = _representation.main_no_tuning_vs_tuning(args)
								res_ = [dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history]
								res.append(res_)
								# avoid losing any result, so save it immediately
								out_file = f'{args.out_dir}/{args.direction}/~{res_file}.dat'
								dump(res, out_file)
								save2txt(res, os.path.splitext(out_file)[0] + '.csv', delimiter=',')
							except Exception as e:
								lg.error(e)

		out_file = f'{args.out_dir}/{args.direction}/{DATE}/{res_file}.dat'

	check_path(out_file)
	dump(res, out_file)
	save2txt(res, os.path.splitext(out_file)[0] + '.csv', delimiter=',')
	lg.info(f'final result: {out_file}')



def gather(in_dir = 'examples/representation/out/src', out_dir = ''):
	res = []
	for dataset, feature, header, model, tuning in list(itertools.product(DATASETS,
	                                                                      FEATURES, HEADER, MODELS, TUNING)):
		f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}', 'res.csv')
		try:
			line = [str(v) for v in pd.read_csv(f, sep=',',header=None).values.flatten().tolist()][1:]
			print(f, line)
		except Exception as e:
			print(f'Error: {e}')
			line = ['', '0_0|0_0|0_0', ''] # [score, shape, params]
		res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}'] + line)

	out_file = os.path.join(out_dir, 'gather.csv')
	check_path(out_file)

	with open(out_file, 'w') as f:
		for vs in res:
			f.write(','.join(vs) + '\n')

	short_file = os.path.join(os.path.split(out_file)[0], 'short.csv')
	# data = pd.read_csv(out_file, error_bad_lines=False, header=None)  # will miss some results.
	# data.iloc[:, 0:7].to_csv(short_file)
	with open(short_file, 'w') as f:
		for vs in res:
			tmp = vs[6].split('|')
			shape = '|'.join(v.split('_')[0] for v in tmp)
			dim = tmp[0].split('_')[1]
			f.write(','.join(vs[:6] + [shape, dim]) + '\n')

	return out_file


@timer
def main_all():
	# # clean()
	# try:
	# 	main()
	# except Exception as e:
	# 	pass
	in_dir ='examples/representation/out/src'
	out_file = gather(in_dir, out_dir = os.path.join(in_dir, str(DATE)))
	print(out_file)
	# out_file = report(in_file = out_file)


if __name__ == '__main__':
	main_all()
