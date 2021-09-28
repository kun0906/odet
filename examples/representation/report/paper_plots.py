""" paper tables

	Instructions:
		# Note: run the below command under the root directory of the project.
		PYTHONPATH=. python3.7 examples/representation/report/paper_plots.py

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger as lg
from matplotlib.colors import ListedColormap

from odet.utils.tool import check_path, timer
######################################################################################################################
### set seaborn background colors
# sns.set_style("darkgrid")
from representation._constants import data_orig2name
from representation.report._base import format_name, parse_csv

sns.set_style("whitegrid")
sns.despine(bottom=True, left=True, top=False, right=False)


def seaborn_palette(feat_type='', fig_type='diff'):
	""" Get colors for difference cases.

	Parameters
	----------
	feat_type
	fig_type

	Returns
	-------

	"""
	# 1 bright used for basic
	# Set the palette to the "pastel" default palette:
	sns.set_palette("bright")
	# sns.palplot(sns.color_palette());
	# plt.show()
	colors_bright = sns.color_palette()

	# muted for FFT
	sns.set_palette("muted")
	colors_muted = sns.color_palette()

	# dark for feature + size
	sns.set_palette("dark")
	colors_dark = sns.color_palette()

	# deep for feature + header
	sns.set_palette("deep")  # for feature+header
	colors_deep = sns.color_palette()

	# colorblind for diff
	sns.set_palette("colorblind")  # for feature+header
	colors_colorblind = sns.color_palette()

	colors_bright = ListedColormap(colors_bright.as_hex()).colors
	colors_dark = ListedColormap(colors_dark.as_hex()).colors
	colors_muted = ListedColormap(colors_muted.as_hex()).colors
	colors_deep = ListedColormap(colors_deep.as_hex()).colors
	colors_colorblind = ListedColormap(colors_colorblind.as_hex()).colors

	feat_type = feat_type.upper()
	fig_type = fig_type.upper()

	C_STATS = 4  # purple
	C_IAT = 2  # green
	C_SIZE = 0  # blue
	C_SAMP_NUM = 3  # red
	C_SAMP_SIZE = 5  # brown

	raw_feat = {'STATS': colors_bright[C_STATS], 'IAT': colors_bright[C_IAT], 'SIZE': colors_bright[C_SIZE],
	            'SAMP-NUM': colors_bright[C_SAMP_NUM],
	            'SAMP-SIZE': colors_bright[C_SAMP_SIZE]}

	if feat_type == "fft_effect".upper():
		if fig_type == 'raw'.upper():
			colors = {'STATS': raw_feat['STATS'], 'IAT': raw_feat['IAT'], 'IAT-FFT': colors_dark[C_IAT],
			          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-NUM-FFT': colors_dark[C_SAMP_NUM]
			          }  # red
		elif fig_type == 'diff'.upper():
			# 'IAT' vs. IAT-FFT
			colors = {'IAT vs. IAT-FFT': raw_feat['IAT'],  # green
			          'SAMP-NUM vs. SAMP-NUM-FFT': raw_feat['SAMP-NUM'],
			          'SAMP-SIZE vs. SAMP-SIZE-FFT': raw_feat['SAMP-SIZE']}
		else:
			msg = f'{feat_type} is not implemented yet.'
			raise ValueError(msg)

	elif feat_type == "size_effect".upper():
		if fig_type == 'raw'.upper():
			colors = {'STATS': raw_feat['STATS'], 'SIZE': raw_feat['SIZE'], 'IAT': raw_feat['IAT'],
			          'IAT+SIZE': colors_dark[C_SIZE],
			          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-SIZE': raw_feat['SAMP-SIZE']
			          }  # red
		elif fig_type == 'diff'.upper():
			colors = {'IAT vs. IAT+SIZE': raw_feat['IAT'],  # green
			          'SAMP-NUM vs. SAMP-SIZE': raw_feat['SAMP-SIZE']}  # red
		else:
			msg = f'{feat_type} is not implemented yet.'
			raise ValueError(msg)
	elif feat_type == "header_effect".upper():
		if fig_type == 'raw'.upper():
			colors = {'STATS (wo. header)': raw_feat['STATS'], 'STATS (w. header)': colors_dark[C_STATS],
			          'IAT+SIZE (wo. header)': colors_dark[C_SIZE], 'IAT+SIZE (w. header)': colors_deep[C_SIZE],
			          # green
			          'SAMP-SIZE (wo. header)': raw_feat['SAMP-SIZE'],
			          'SAMP-SIZE (w. header)': colors_deep[C_SAMP_SIZE]}  # red
		elif fig_type == 'diff'.upper():
			colors = {'STATS (wo. header) vs. STATS (w. header)': raw_feat['STATS'],
			          'IAT+SIZE (wo. header) vs. IAT+SIZE (w. header)': raw_feat['IAT'],  # green
			          'SAMP-SIZE (wo. header) vs. SAMP-SIZE (w. header)': raw_feat['SAMP-SIZE']}  # red
		else:
			msg = f'{feat_type} is not implemented yet.'
			raise ValueError(msg)

	else:
		msg = f'{feat_type} is not implemented yet.'
		raise ValueError(msg)

	return colors


def get_ylim(ys):
	""" Get y ticks based on ys

	Parameters
	----------
	ys: all y values

	Returns
	-------

	"""

	max_v = max(ys)
	min_v = min(ys)

	# find the closest value to min_v
	if min_v == 0:
		pass
	elif min_v > 0:
		vs = [0, 0.25, 0.5, 0.75, 1.0]
		v = min(vs, key=lambda x: abs(x - min_v))
		min_v = v - 0.25 if min_v < v else v
	else:
		vs = [0, -0.25, -0.5, -0.75, -1.0]
		v = min(vs, key=lambda x: abs(x - min_v))
		min_v = v - 0.25 if min_v < v else v

	# find the closest value to max_V
	if max_v == 0:
		pass
	elif max_v > 0:
		vs = [0, 0.25, 0.5, 0.75, 1.0]
		v = min(vs, key=lambda x: abs(x - max_v))
		max_v = v + 0.25 if max_v > v else v
	else:
		vs = [0, -0.25, -0.5, -0.75, -1.0]
		v = min(vs, key=lambda x: abs(x - max_v))
		max_v = v + 0.25 if max_v > v else v

	v = min_v
	ys = []
	# k = (max_v - min_v) // 0.25 + 1
	while v <= max_v:
		ys.append(v)
		v += 0.25

	while len(ys) <= 3:
		min_v = min_v - 0.25
		max_v = max_v + 0.25
		ys = [min_v] + ys + [max_v]

	return min_v, max_v, ys


def get_yticklabels(ys):
	""" get y labels based on ys

	Parameters
	----------
	ys: all y values

	Returns
	-------

	"""
	# if len(_ys_fmt) >=6:
	#     _ys_fmt = [ f'{_v:.2f}' if _i % 2 == 0 or _v == 0 or _i == len(_ys_fmt) - 1 else '' for _i, _v in
	#     enumerate(_ys_fmt) ]
	# else:
	#     _ys_fmt = [f'{_v:.2f}' for _v in _ys_fmt]

	ys_fmt = [''] * len(ys)  # y labels
	ys_fmt[0] = ys[0]
	ys_fmt[-1] = ys[-1]

	### find '0' position
	i = 0
	for i, v in enumerate(ys):
		if v == 0:
			ys_fmt[i] = v
			break

	if ys[0] == -1.0:
		ys_fmt[2] = -0.5
	elif ys[0] == -0.75:
		ys_fmt[1] = -0.5
	# elif ys[0] == -0.5:
	#     ys_fmt[1] = -0.25

	if ys[-1] == 1.0:
		ys_fmt[-3] = 0.5
	elif ys[-1] == 0.75:
		ys_fmt[-2] = 0.5
	elif ys[-1] == 0.5 and ys[0] == -0.5:
		ys_fmt[-2] = 0.25
		ys_fmt[1] = -0.25

	# special cases:
	if ys[0] == -0.25:
		if ys[-1] == 0.75 or ys[-1] == 0.5:
			ys_fmt[2] = 0.25
	elif ys[-1] == 0.25:
		if ys[0] == -0.75 or ys[0] == -0.5:
			ys_fmt[-3] = -0.25

	return [f'{v:.2f}' if v != '' else '' for v in ys_fmt]


def bar_plot(data, MODELS=[], fig_type='', out_file="F1_for_all.pdf", n_rows=3, n_cols=2):
	"""

	Parameters
	----------
	data
	MODELS
	fig_type
	out_file
	n_rows
	n_cols

	Returns
	-------

	"""
	colors = seaborn_palette(fig_type, fig_type='diff').values()
	if n_cols == 1:
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 25))  # (width, height)
		axes = axes.reshape(n_rows, -1)
	else:
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))  # (width, height)

	# [tuning, model, dataset,'IAT+SIZE\\IAT', d, s])
	df = pd.DataFrame(data, columns=['tuning', 'model', 'dataset', 'feature', 'diff', 'std_error'])
	for r in range(n_rows):
		for c in range(n_cols):
			model = MODELS[r * n_cols + c]
			_y_min, _y_max, _ys = get_ylim(df['diff'].values)
			lg.debug(f'{r}, {c}, {model}, {_ys}')
			_df = df[df['model'] == model]
			g = sns.barplot(y="diff", x='dataset', hue='feature', data=_df, palette=colors, ci=None,
			                capsize=.2, ax=axes[r, c])

			# compute the error bar position
			ys = []
			xs = []
			yerrs = _df['std_error'].values
			ns_ = [int(1 / v ** 2) for v in yerrs]
			ds_ = _df['dataset'].values.tolist()
			lg.debug(f'{list(zip(yerrs, ns_, ds_))}')

			# rearrange yerrs
			yerrs_ = []
			num_bars = _df['feature'].nunique()
			for i in range(num_bars):
				yerrs_.extend(yerrs[i::num_bars])
			yerrs = yerrs_
			for i_p, p in enumerate(g.patches):
				height = p.get_height()
				width = p.get_width()
				ys.append(height)
				xs.append(p.get_x())
				num_bars = _df['feature'].nunique()

				if i_p == 0:
					pre = p.get_x() + p.get_width() * num_bars
				# sub_fig_width = p.get_bbox().width
				if i_p < _df['dataset'].nunique() and i_p > 0:
					cur = p.get_x()
					g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
					pre = cur + p.get_width() * num_bars

			axes[r, c].errorbar(x=xs + width / 2, y=ys, yerr=yerrs, fmt='none', ecolor='b', capsize=3)
			g.set(xlabel=None)
			g.set(ylabel=None)

			font_size = 20
			g.set_ylabel('AUC difference', fontsize=font_size + 4)
			# only the last row shows the xlabel.
			if n_rows > 1:
				if r < n_rows - 1:
					g.set_xticklabels([])
				else:
					g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")

			g.set_ylim(_y_min, _y_max)
			# set the yticks and labels
			g.set_yticks(_ys)
			_ys_fmt = get_yticklabels(_ys)
			g.set_yticklabels(_ys_fmt, fontsize=font_size + 6)
			# lg.debug(g.get_yticks(), _ys)

			### don't show the y ticks and labels for rest of columns.
			if c != 0:
				# g.get_yaxis().set_visible(False)
				g.set_yticklabels(['' for v_tmp in _ys])
				g.set_ylabel('')

			# set title for each subplot
			g.set_title(model, fontsize=font_size + 8)
			g.get_legend().set_visible(False)

	### get the legend from the last 'ax' (here is 'g') and relocated its position.
	handles, labels = g.get_legend_handles_labels()
	labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
	fig.legend(handles, labels, loc='lower center', ncol=3, prop={'size': font_size - 2}, frameon=False)

	plt.tight_layout()
	if n_cols == 2:
		plt.subplots_adjust(bottom=0.17)
	else:
		plt.subplots_adjust(bottom=0.07)

	# should use before plt.show()
	plt.savefig(out_file)
	# plt.show()
	plt.close(fig)
	lg.debug(out_file)


def size_effect_plot(data, MODELS=[], DATASETS=[], fig_type='', out_file="F1_for_all.pdf"):
	"""

	Parameters
	----------
	data
	MODELS
	DATASETS
	fig_type
	out_file

	Returns
	-------

	"""
	if len(DATASETS) > 10:
		n_rows = 6
		n_cols = 1
	else:
		n_rows = 3
		n_cols = 2
	bar_plot(data, MODELS, fig_type, out_file, n_rows, n_cols)


def header_effect_plot(data, MODELS=[], DATASETS=[], fig_type='', out_file="F1_for_all.pdf"):
	"""

	Parameters
	----------
	data
	MODELS
	DATASETS
	fig_type
	out_file

	Returns
	-------

	"""
	if len(DATASETS) > 10:
		n_rows = 6
		n_cols = 1
	else:
		n_rows = 3
		n_cols = 2
	bar_plot(data, MODELS, fig_type, out_file, n_rows, n_cols)


def fft_effect_plot(data, MODELS=[], DATASETS=[], fig_type='', out_file="F1_for_all.pdf"):
	"""

	Parameters
	----------
	data
	MODELS
	DATASETS
	fig_type
	out_file

	Returns
	-------

	"""
	if len(DATASETS) > 10:
		n_rows = 6
		n_cols = 1
	else:
		n_rows = 3
		n_cols = 2
	bar_plot(data, MODELS, fig_type, out_file, n_rows, n_cols)


def pkt_size_diff(data, tuning, DATASETS, MODELS):
	"""

	Parameters
	----------
	data
	tuning
	DATASETS
	MODELS

	Returns
	-------

	"""

	def _diff(data, header, tuning, f1, f2, dataset, model):
		try:
			score1, shape1, dim1 = data[header][tuning][f1][dataset][model]
			score2, shape2, dim2 = data[header][tuning][f2][dataset][model]
			n_test = int(shape1.split('|')[-1])
			diff = float(score1) - float(score2)
			std_error = 1 / np.sqrt(n_test)
		except Exception as e:
			lg.error(f'Error: {e}. {header}, {tuning}, {f1}, {f2}, {dataset}, {model}')
			# traceback.print_exc()
			diff = 0
			std_error = 0
		return diff, std_error

	results = []
	tuning = f'tuning_{tuning}'
	for dataset in DATASETS:
		for model in MODELS:
			header = 'header_False'
			# IAT+SIZE - IAT
			d, s = _diff(data, header, tuning, 'IAT+SIZE', 'IAT', dataset, model)
			results.append([tuning, model, dataset, 'IAT+SIZE \\ IAT', d, s])
			# SAMP_SIZE - SAMP_NUM
			d, s = _diff(data, header, tuning, 'SAMP_SIZE', 'SAMP_NUM', dataset, model)
			results.append([tuning, model, dataset, 'SAMP-SIZE \\ SAMP-NUM', d, s])

	return results


def pkt_header_diff(data, tuning, DATASETS, MODELS):
	"""

	Parameters
	----------
	data
	tuning
	DATASETS
	MODELS

	Returns
	-------

	"""

	def _diff(data, h1, h2, tuning, feature, dataset, model):
		try:
			score1, shape1, dim1 = data[h1][tuning][feature][dataset][model]
			score2, shape2, dim2 = data[h2][tuning][feature][dataset][model]
			n_test = int(shape1.split('|')[-1])
			diff = float(score1) - float(score2)
			std_error = 1 / np.sqrt(n_test)
		except Exception as e:
			lg.error(f'Error: {e}. {h1}, {h2}, {tuning}, {feature}, {dataset}, {model}')
			diff = 0
			std_error = 0
		return diff, std_error

	results = []
	tuning = f'tuning_{tuning}'
	for dataset in DATASETS:
		for model in MODELS:
			# STATS(w.header) - STATS(wo. header)
			d, s = _diff(data, 'header_True', 'header_False', tuning, 'STATS', dataset, model)
			feat = 'STATS (w.header) \\ (wo. header)'
			results.append([tuning, model, dataset, feat, d, s])
			# IAT+SIZE(w.header) - IAT+SIZE(wo. header)
			d, s = _diff(data, 'header_True', 'header_False', tuning, 'IAT+SIZE', dataset, model)
			feat = 'IAT+SIZE (w.header) \\ (wo. header)'
			results.append([tuning, model, dataset, feat, d, s])
			# SAMP_SIZE(w.header) - SAMP_SIZE(wo. header)
			d, s = _diff(data, 'header_True', 'header_False', tuning, 'SAMP_SIZE', dataset, model)
			feat = 'SAMP-SIZE(w.header) \\ (wo. header)'
			results.append([tuning, model, dataset, feat, d, s])

	return results


def fft_diff(data, tuning, DATASETS, MODELS):
	"""

	Parameters
	----------
	data
	tuning
	DATASETS
	MODELS

	Returns
	-------

	"""

	def _diff(data, header, tuning, f1, f2, dataset, model):
		try:
			score1, shape1, dim1 = data[header][tuning][f1][dataset][model]
			score2, shape2, dim2 = data[header][tuning][f2][dataset][model]
			n_test = int(shape1.split('|')[-1])
			diff = float(score1) - float(score2)
			std_error = 1 / np.sqrt(n_test)
		except Exception as e:
			lg.error(f'Error: {e}. {header}, {tuning}, {f1}, {f2}, {dataset}, {model}')
			diff = 0
			std_error = 0
		return diff, std_error

	results = []
	tuning = f'tuning_{tuning}'
	for dataset in DATASETS:
		for model in MODELS:
			header = 'header_False'
			# 'FFT_IAT'- 'IAT'
			d, s = _diff(data, header, tuning, 'FFT_IAT', 'IAT', dataset, model)
			feat = 'IAT-FFT \\ IAT'
			results.append([tuning, model, dataset, feat, d, s])
			d, s = _diff(data, header, tuning, 'FFT_SAMP_NUM', 'SAMP_NUM', dataset, model)
			feat = 'SAMP-NUM-FFT \\ SAMP-NUM'
			results.append([tuning, model, dataset, feat, d, s])
			d, s = _diff(data, header, tuning, 'FFT_SAMP_SIZE', 'SAMP_SIZE', dataset, model)
			feat = 'SAMP-SIZE-FFT \\ SAMP-SIZE'
			results.append([tuning, model, dataset, feat, d, s])

	return results


@timer
def main():
	"""Get results from xlsx and plot the results.

	Parameters
	----------
	root_dir

	Returns
	-------

	"""
	# raw_file = 'examples/representation/out/src/~res.csv'
	# in_file = 'examples/representation/report/res.csv'
	# check_path(in_file)
	# copyfile(raw_file, in_file)
	in_file = 'examples/representation/out/src/results/2021-09-28/short.csv'
	data = parse_csv(in_file)
	data = format_name(data, data_orig2name)  #
	out_dir = 'examples/representation/report/out'
	TUNING = [True, False]
	MODELS = ['OCSVM', 'IF', 'AE', 'KDE', 'GMM', 'PCA']

	########################################################################################################
	### Get the results on part of datasets and all algorithms
	# 1. datasets: [UNB(PC1), UNB(PC4), CTU, MAWI, TV&RT, SFrig, and BSTch]
	# algorithms: [OCSVM, IF, AE, KDE, GMM, PCA]
	DATASETS1 = ['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig','BSTch']
	DATASETS2 = ['UNB(PC2)', 'UNB(PC3)', 'UNB(PC5)', 'SCam', 'GHom']
	DATASETS3 = ['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch',
	             'SCam', 'GHom']
	DATASETS_LST = [DATASETS1, DATASETS2, DATASETS3]
	FIGS = ['size_effect', 'header_effect', 'fft_effect']
	for tuning in TUNING:
		tuning_type = 'best' if tuning else 'default'
		for DATASETS in DATASETS_LST:
			# 1. size_effect
			fig_type = FIGS[0]
			out_file = f'{out_dir}/{fig_type}-{tuning_type}-{len(DATASETS)}.pdf'
			check_path(out_file)
			results = pkt_size_diff(data, tuning, DATASETS, MODELS)
			size_effect_plot(results, MODELS, DATASETS, fig_type, out_file)

			# 2. header_effect
			fig_type = FIGS[1]
			out_file = f'{out_dir}/{fig_type}-{tuning_type}-{len(DATASETS)}.pdf'
			check_path(out_file)
			results = pkt_header_diff(data, tuning, DATASETS, MODELS)
			header_effect_plot(results, MODELS, DATASETS, fig_type, out_file)

			# 3. fft_effect
			fig_type = FIGS[2]
			out_file = f'{out_dir}/{fig_type}-{tuning_type}-{len(DATASETS)}.pdf'
			check_path(out_file)
			results = fft_diff(data, tuning, DATASETS, MODELS)
			fft_effect_plot(results, MODELS, DATASETS, fig_type, out_file)

			lg.info('\n')


if __name__ == '__main__':
	main()
