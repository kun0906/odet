""" paper tables

	Instructions:
		# Note: run the below command under the root directory of the project.
		PYTHONPATH=. python3.7 examples/representation/report/paper_tables.py

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import os

from loguru import logger as lg

from odet.utils.tool import timer
from representation._constants import data_orig2name
from representation.report.paper_plots import parse_csv, format_name


def form_dim_table(data, header, tuning, features=[], datasets=[], model=''):
	"""form dataset table in the paper

	Parameters
	----------
	data
	header
	tuning
	features
	datasets
	models

	Returns
	-------

	"""
	result = []
	result.append(['Device'] + features)
	for dataset in datasets:
		vs = []
		for feature in features:
			try:
				tmp = data[header][tuning][feature][dataset][model]  # (score, shape, dim)
				tmp = f'{tmp[-1].strip()}'
			except Exception as e:
				lg.error(f'Error: {e}. [{header}, {tuning}, {feature}, {dataset}, {model}, {tmp}]')
				tmp = ''
			vs.append(tmp)
		line = [dataset] + vs
		result.append(line)

	return result


def form_dataset_table(data, header, tuning, feature=[], datasets=[], model=''):
	"""form dataset table in the paper

	Parameters
	----------
	data
	header
	tuning
	features
	datasets
	models

	Returns
	-------

	"""
	result = []
	result.append(['Reference', 'Devices', 'Train set', 'Validation set', 'Test set'])
	for dataset in datasets:
		try:
			tmp = data[header][tuning][feature][dataset][model]  # (score, shape, dim)
			tmp = tmp[1].split('|')
			train = tmp[0]
			val = int(int(tmp[1]) // 2)
			test = int(int(tmp[2]) // 2)
			vs = [f'N:{train}', f'N:{val} A:{val}', f'N:{test} A:{test}']
		except Exception as e:
			lg.error(f'Error: {e}. [{header}, {tuning}, {feature}, {dataset}, {model}, {vs}]')
			vs = []
		line = [dataset, dataset] + vs
		result.append(line)

	return result


def form_score_table(data, header, tuning, features, datasets, models):
	"""form tables in the paper

	Parameters
	----------
	data
	header
	tuning
	features
	datasets
	models

	Returns
	-------

	"""
	result = []
	result.append(['Detector', 'Dataset'] + features)
	for model in models:
		for dataset in datasets:
			vs = []
			for feature in features:
				try:
					tmp = data[header][tuning][feature][dataset][model]  # (score, shape, dim)
					tmp = f'{float(tmp[0]):.2f}'
				except Exception as e:
					lg.error(f'Error: {e}. [{header}, {tuning}, {feature}, {dataset}, {model}, {tmp}]')
					tmp = ''
				vs.append(tmp)
			line = [model, dataset] + vs
			result.append(line)

	return result


def score2latex(data, datasets, out_file=''):
	"""

	\multirow{7}{*}{OCSVM} &UNB(PC1) & 0.48 & 0.77 & 0.81 & 0.76   \\
	&UNB(PC4) & 0.55 & 0.73 & 0.86 &  0.75  \\
	\cmidrule{2-6}
	&CTU & 0.64 & 0.77 & 0.76 & 0.91\\
	\cmidrule{2-6}
	&MAWI & 0.88 & 0.47 & 0.45 & 0.59\\
	\cmidrule{2-6}
	&TV\&RT & 1.00 & 1.00 & 0.96 & 0.93 \\
	\cmidrule{2-6}
	&SFrig & 0.98 & 0.89 & 0.82  & 0.96  \\
	&BSTch & 0.95 & 0.98 & 0.97 & 0.95 \\
	\midrule
	Parameters
	----------
	result

	Returns
	-------

	"""

	results = []
	pre = 'Detector'
	for vs in data:
		if pre == vs[0]:
			line = ' & ' + ' & '.join(vs[1:]) + ' \\\\'
			results.append(line.replace('TV&RT', 'TV\\&RT'))
			results.append('\\cmidrule{2-6}')
		else:
			results[-1] = '\\midrule'
			pre = vs[0]
			line = '\multirow{' + str(len(datasets)) + '}{*}{' + pre + '} & ' + ' & '.join(vs[1:]) + ' \\\\'
			results.append(line.replace('TV&RT', 'TV\\&RT'))

	with open(out_file, 'w') as f:
		for vs in results:
			f.write(vs + '\n')


def save2txt(data, out_file):
	with open(out_file, 'w') as f:
		for vs in data:
			line = ','.join(vs)
			f.write(line + '\n')


@timer
def main():
	"""

	Parameters
	----------
	root_dir

	Returns
	-------

	"""
	in_file = 'examples/representation/out/src/results/20210912/short.csv'
	in_file = 'examples/representation/out/src/results/20210913/short.csv'
	data = parse_csv(in_file)
	data = format_name(data, data_orig2name)  #
	out_dir = 'examples/representation/report/out'
	TUNING = [True, False]
	MODELS = ['OCSVM', 'IF', 'AE', 'KDE', 'GMM', 'PCA']
	FEATURES = ['STATS', 'SIZE', 'IAT', 'SAMP_NUM']

	DATASETS1 = ['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']
	DATASETS2 = ['UNB(PC2)', 'UNB(PC3)', 'UNB(PC5)', 'SCam', 'GHom']
	DATASETS3 = ['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)', 'CTU', 'MAWI', 'TV&RT',  'SFrig', 'BSTch',
	             'SCam','GHom']
	# 1. get scores
	for datasets in [DATASETS1, DATASETS2, DATASETS3]:
		for tuning in TUNING:
			tuning = f'tuning_{tuning}'
			result = form_score_table(data, 'header_False', tuning, FEATURES, datasets, MODELS)
			out_file = os.path.join(out_dir, f'{tuning}-{len(datasets)}.csv')
			save2txt(result, out_file)
			score2latex(result, datasets, os.path.splitext(out_file)[0] + '-latax.txt')
			lg.info(f'{tuning}, {datasets}, {out_file}')

	# 2. get dimensions of features
	header = 'header_False'
	tuning = 'tuning_False'
	model = 'PCA'
	FEATURES = ['IAT', 'SIZE', 'STATS', 'SAMP_NUM']
	result = form_dim_table(data, header, tuning, FEATURES, DATASETS3, model)
	out_file = os.path.join(out_dir, f'dimensions.csv')
	save2txt(result, out_file)
	lg.info(f'{tuning}, {datasets}, {out_file}')

	# 3. get train, val and test set
	result = form_dataset_table(data, header, tuning, 'IAT', DATASETS3, model)
	out_file = os.path.join(out_dir, f'datasets.csv')
	save2txt(result, out_file)
	lg.info(f'{tuning}, {datasets}, {out_file}')


if __name__ == '__main__':
	main()
