""" Gather all results:

	Instructions:
		# Note: run the below command under the root directory of the project.
		python3.7 examples/representation/_gather.py

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
from examples.representation._constants import *  # should in the top.
import itertools

import pandas as pd
from odet.utils.tool import timer, check_path


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


if __name__ == '__main__':
	OUT_DIR = r'examples/representation/out'
	in_dir = f'{OUT_DIR}/src'
	out_dir = os.path.join(in_dir, f'results/{START_TIME}')
	out_file = gather(in_dir, out_dir)
	lg.info(out_file)
