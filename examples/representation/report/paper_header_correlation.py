"""Get correlation

    Run in the terminal:
        PYTHONPATH=. python3.7 examples/representation/report/paper_header_correlation.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from examples.representation._constants import *  # should in the top.
from odet.utils.tool import load, check_path, dump, split_train_val_test, normalize

######################################################################################################################
### add current path to system path
lib_path = os.path.abspath('.')
sys.path.append(lib_path)
# print(f"add \'{lib_path}\' into sys.path: {sys.path}")

######################################################################################################################
### set seaborn background colors
# sns.set_style("darkgrid")
sns.set_style("whitegrid")
sns.despine(bottom=True, left=True, top=False, right=False)
RANDOM_STATE = 42


def _get_each_correlation(x, y):
	lg.debug(f'{np.std(x)}, {np.std(y)}')
	rho = np.corrcoef(x, y)[0, 1]
	rho = 0 if np.isnan(rho) else rho
	return rho


def get_correlation(in_dir='',
                    datasets='',
                    feature='SIZE',
                    header=True,
                    out_dir='',
                    out_file='.dat'):
	corr_results = {}
	for i, dataset in enumerate(datasets):
		in_file = os.path.join(in_dir, dataset, feature, f"header_{header}", 'Xy.dat')
		lg.debug(in_file)
		data = load(in_file)
		X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(data['X'], data['y'],
		                                                                      shuffle=True,
		                                                                      random_state=RANDOM_STATE)
		# normalization
		ss, X_train, y_train, X_val, y_val, X_test, y_test = normalize(X_train, y_train, X_val, y_val, X_test,
		                                                               y_test)
		# 2 get correlation
		dim = X_test.shape[1]
		if feature == 'IAT':
			# iat_dim + header_dim = dim, here header_dim =  (8 + ttl_dim (i.e., size_dim))
			# => iat_dim + 8 + size_dim = iat_dim + 8 + (iat_dim + 1) = dim
			# => iat_dim = (dim - 9)//2
			start_idx = (dim - 8 - 1) // 2
		elif feature == 'SIZE':
			# size_dim + header_dim = dim
			# size_dim + (8+size_dim) = dim
			# size_dim = (dim - 8 ) // 2
			start_idx = (dim - 8) // 2  # # feature + header_feature:(8 tcp flags + TTL). only works for 'SIZE'
		else:
			msg = f'Error: {feature}'
			raise NotImplementedError(msg)
		corrs = []
		lg.debug(f'header_feature_start_idx: {start_idx}')
		for j in range(9):  # feature + header_feature:(8 tcp flags + first TTL)
			_corr = _get_each_correlation(X_test[:, start_idx + j], y_test)
			corrs.append(_corr)
		corr_results[(in_file, dataset, feature, X_test.shape)] = corrs

		_out_file = os.path.join(out_dir, dataset, 'correlation.dat')
		check_path(_out_file)
		dump(corrs, _out_file)
		print(_out_file)
	# save all results
	check_path(out_file)
	dump(corr_results, out_file)

	return out_file


def plot_correlation_multi(corr_results, out_file='', title=None, show=True):
	""" plot the data

	Parameters
	----------
	corr_results
	out_dir
	title
	show

	Returns
	-------

	"""
	# # only show the top 4 figures
	new_corr_results = {}
	for i, (dataset, name) in enumerate(data_orig2name.items()):
		for j, (key, corrs) in enumerate(corr_results.items()):
			_key_path, _dataset, _feat_set, X_test_shape = key
			if dataset in key:
				new_corr_results[(_key_path, _dataset, name, _feat_set, X_test_shape)] = corrs
	t = 0
	cols = 2
	fontsize = 20
	## http://jose-coto.com/styling-with-seaborn
	# colors = ["m", "#4374B3"]
	# palette = sns.color_palette('RdPu', 1)  # a list
	palette = [sns.color_palette('YlOrRd', 7)[4]]  # YlOrRd
	fig, axes = plt.subplots(2, cols, figsize=(18, 8))  # (width, height)
	# print(new_corr_results)
	for i, (key, corrs) in enumerate(new_corr_results.items()):
		print(f"i: {i}, {key}, corrs: {corrs}")  # hue = feat_set
		key_path, dataset, short_name, feat_set, X_test_shape = key
		HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']

		data = sorted(range(len(corrs)), key=lambda i: abs(corrs[i]), reverse=True)[:6]  # top 6 values
		data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in data]
		# print(f"i: {i}, {key}, corrs: {data}")

		new_yerrs = [1 / (np.sqrt(X_test_shape[0]))] * 6  # for the same dataset, it has the same err_bar
		# # print(f'i: {i}, {new_yerrs}')

		df = pd.DataFrame(data, columns=[f'Xi_y', 'feat_set', 'corr_rho'])
		if i % cols == 0 and i > 0:
			t += 1
		g = sns.barplot(x=f"Xi_y", y="corr_rho", ax=axes[t, i % cols], hue='feat_set', data=df,
		                palette=palette)  # palette=palette,
		g.set(xlabel=None)
		g.set(ylim=(-1, 1))
		if i % cols == 0:
			# g.set_ylabel(r'$\rho$', fontsize=fontsize + 4)
			g.set_ylabel(r'Correlation', fontsize=fontsize + 4)
			# print(g.get_yticks())
			g.set_yticks([-1, -0.5, 0, 0.5, 1])
			g.set_yticklabels(g.get_yticks(), fontsize=fontsize + 6)  # set the number of each value in y axis
		# print(g.get_yticks())
		else:
			g.set(ylabel=None)
			g.set_yticklabels(['' for v_tmp in g.get_yticks()])
			g.set_ylabel('')

		# g.set_title(dataset_name)
		g.get_legend().set_visible(False)
		g.set_xticklabels(g.get_xticklabels(), fontsize=fontsize + 4, rotation=30, ha="center")

		ys = []
		xs = []
		width = 0
		for i_p, p in enumerate(g.patches):
			height = p.get_height()
			width = p.get_width()
			ys.append(height)
			xs.append(p.get_x())
			if i_p == 0:
				pre = p.get_x() + p.get_width()
			if i_p > 0:
				cur = p.get_x()
				g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
				pre = cur + p.get_width()
			## https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
			p.set_width(width / 3)  # set the bar width
			# we recenter the bar
			p.set_x(p.get_x() + width / 3)
		g.set_title(short_name, fontsize=fontsize + 8)

		# add error bars
		g.errorbar(x=xs + width / 2, y=ys,
		           yerr=new_yerrs, fmt='none', c='b', capsize=3)

	# # get the legend and modify it
	# handles, labels = g.get_legend_handles_labels()
	# fig.legend(handles, ['IAT+SIZE'], title=None, loc='lower center', ncol=1,
	#  prop={'size': fontsize-2})  # loc='lower right',  loc = (0.74, 0.13)

	plt.tight_layout()
	plt.subplots_adjust(bottom=0.2)

	check_path(out_file)
	print(out_file)
	plt.savefig(out_file)  # should use before plt.show()
	if show: plt.show()
	plt.close(fig)
	plt.close("all")


def main():
	root_dir = 'examples/representation'
	in_dir = f'{root_dir}/report/out/'
	corr_file = os.path.join(in_dir, 'correlation', 'correlation.dat')
	DATASETS = ['UNB(PC1)',
	            'CTU',
	            'MAWI',
	            'UCHI(SFRIG_2021)']
	feature = 'SIZE'  # the correlation code only works for SIZE
	header = True
	out_dir = f'{root_dir}/report/out/correlation'
	get_correlation(in_dir=f'{root_dir}/out/src',
	                datasets=DATASETS,
	                feature=feature,
	                header=header,
	                out_dir=out_dir,
	                out_file=corr_file)
	data = load(corr_file)
	out_file = os.path.join(out_dir, feature, f"header_{header}", 'correlation.pdf')
	plot_correlation_multi(data, out_file=out_file, show=True)


if __name__ == '__main__':
	main()
