"""
     for both MAWI, SFrig and UNB(PC1) ?
- Let Y in {0, 1} denote normal or novelty, and X1, .., X9 denote the packet header information.
- Compute the correlation(Xi, Y) for each of the Xi’s (do this on the test data).

    command



"""
import os, sys
import pickle
from collections import Counter

import sklearn
from sklearn.model_selection import train_test_split

lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set()


# from itod.ndm.train_test import extract_data, preprocess_data
# from itod.utils.utils import dump_data


def dump_data(data, file_out='data.dat'):
    with open(file_out, 'wb') as f:
        pickle.dump(data, f)


# 0.4 get root logger
# from itod import log

# lg = log.get_logger(name=None, level='DEBUG', out_dir='./log/data_kjl', app_name='correlation')

datasets = [
    # Naming: (department/dataname_year/device)
    'UNB/CICIDS_2017/pc_192.168.10.5',
    # 'UNB/CICIDS_2017/pc_192.168.10.8',
    # 'UNB/CICIDS_2017/pc_192.168.10.9',
    # 'UNB/CICIDS_2017/pc_192.168.10.14',
    # 'UNB/CICIDS_2017/pc_192.168.10.15',
    # #
    'CTU/IOT_2017/pc_10.0.2.15',
    #
    'MAWI/WIDE_2019/pc_202.171.168.50',
    # 'MAWI/WIDE_2020/pc_203.78.7.165',
    # # # # #
    # 'UCHI/IOT_2019/smtv_10.42.0.1',
    #
    # 'UCHI/IOT_2019/ghome_192.168.143.20',
    # 'UCHI/IOT_2019/scam_192.168.143.42',
    'UCHI/IOT_2019/sfrig_192.168.143.43',
    # 'UCHI/IOT_2019/bstch_192.168.143.48'

]


def _get_each_correlation(x, y):
    rho = np.corrcoef(x, y)[0, 1]
    rho = 0 if np.isnan(rho) else rho
    return rho


def get_data(in_file='.dat', feat_set='iat_size'):
    with open(in_file, 'rb') as f:
        inst = pickle.load(f)
    X_train, y_train, X_test, y_test = inst.dataset_inst.dataset_dict[f'{feat_set}_dict']['data']

    size = 5000
    if len(y_train) > size:
        X_train, y_train = sklearn.utils.resample(X_train, y_train, n_samples=size, random_state=42, replace=False)

    # sd_inst = Dataset()
    # X_train, X_test = sd_inst.normalise_data(X_train, X_test,
    #                                          norm_method=self.params['norm_method'])
    # select a part of val_data
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
                                                    stratify=y_test,
                                                    random_state=42)
    print(f'X_train: {X_train.shape}, y_train: {Counter(y_train)}')
    print(f'X_val: {X_val.shape}, y_val: {Counter(y_val)}')
    print(f'X_test: {X_test.shape}, y_test: {Counter(y_test)}')

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_correlation(in_dir='', out_dir='out', header=True):
    # datasets = ['DS10_UNB_IDS/DS11-srcIP_192.168.10.5', # UNB(PC1)
    #             'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50', # MAWI
    #             'DS60_UChi_IoT/DS63-srcIP_192.168.143.43'   # SFrig
    #             ]

    feat_sets = ['iat_size']  # all the features has the same header

    corr_results = {}
    for i, dataset in enumerate(datasets):
        for feat_set in feat_sets:
            key_pth = os.path.join(in_dir, dataset, feat_set, f"header:{header}")
            print(f'i: {i}, key_path: {key_pth}')
            # 1. get data
            # normal_file = os.path.join(key_pth, 'normal.csv')
            # abnormal_file = os.path.join(key_pth, 'abnormal.csv')
            # print(f'normal_file: {normal_file}')
            # print(f'abnormal_file: {abnormal_file}')
            # meta_data = {'idxs_feat': [0, -1], 'train_size': -1, 'test_size': -1}
            # normal_data, abnormal_data = extract_data(normal_file, abnormal_file, meta_data)
            # X_train, X_test, y_train, y_test, kjl_train_set_time, kjl_test_set_time = preprocess_data(normal_data,
            #                                                                                           abnormal_data,
            #                                                                                           kjl=False, d=0,
            #                                                                                           n=0, quant=0,
            #                                                                                           model_name=None,
            #                                                                                           random_state=42)
            in_file = os.path.join(in_dir, dataset, f'all-features-header:{header}.dat')
            X_train, y_train, X_val, y_val, X_test, y_test = get_data(in_file, feat_set)
            # 2 get correlation
            corrs = []
            for j in range(9):  # the first 9 columns: 8 tcp flags + 1 TTL
                _corr = _get_each_correlation(X_test[:, j], y_test)
                corrs.append(_corr)
            corr_results[(key_pth, dataset, feat_set, X_test.shape)] = corrs

        out_file = os.path.join(out_dir, dataset, feat_set, f"header:{header}", 'correlation.dat')
        print(f'i: {i}, {dataset}, out_file: {out_file}')
        if not os.path.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
        dump_data((key_pth, corrs), out_file)

    # save all results
    out_file = os.path.splitext(out_file)[0] + '_all.dat'
    print(f'out_file: {out_file}')
    dump_data(corr_results, out_file)

    return corr_results


def plot_correlation_multi(corr_results, out_dir, title=None, show=True):
    # # only show the top 4 figures
    datasets = [
        ('UNB/CICIDS_2017/pc_192.168.10.5', 'UNB(PC1)'),  # data_name is unique
        # ('DS10_UNB_IDS/DS14-srcIP_192.168.10.14', 'UNB(PC4)'),
        ('CTU/IOT_2017/pc_10.0.2.15', 'CTU'),
        ('MAWI/WIDE_2019/pc_202.171.168.50', 'MAWI'),
        ('UCHI/IOT_2019/sfrig_192.168.143.43', 'SFrig'),
    ]
    new_corr_results = {}
    for i, (dataset, name) in enumerate(datasets):
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
    print(new_corr_results)
    for i, (key, corrs) in enumerate(new_corr_results.items()):
        print(f"i: {i}, {key}, corrs: {corrs}")  # hue = feat_set
        key_path, dataset, short_name, feat_set, X_test_shape = key
        # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
        # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
        HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']

        data = sorted(range(len(corrs)), key=lambda i: abs(corrs[i]), reverse=True)[:6]  # top 6 values
        # data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in sorted(data, reverse=False)]
        data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in data]
        print(f"i: {i}, {key}, corrs: {data}")

        new_yerrs = [1 / (np.sqrt(X_test_shape[0]))] * 6  # for err_bar
        print(f'i: {i}, {new_yerrs}')
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
            print(g.get_yticks())
            g.set_yticks([-1, -0.5, 0, 0.5, 1])
            g.set_yticklabels(g.get_yticks(), fontsize=fontsize + 6)  # set the number of each value in y axis
            print(g.get_yticks())
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

    # title = dataset
    # # fig.suptitle("\n".join(["a big long suptitle that runs into the title"]*2), y=0.98)
    # fig.suptitle(title)
    out_file = os.path.join(out_dir, feat_set, "header:True", 'correlation-bar.pdf')
    if not os.path.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
    print(out_file)
    plt.savefig(out_file)  # should use before plt.show()
    if show: plt.show()
    plt.close(fig)
    plt.close("all")
    # sns.reset_orig()
    # sns.reset_defaults()


def plot_correlation(corr_results, out_dir, title=None, show=True):
    for i, (key, corrs) in enumerate(corr_results.items()):
        print(f"i: {i}, {key}")  # hue = feat_set
        key_path, dataset, feat_set, X_test_shape = key
        # data = [[f'X{_i+1}_y', feat_set, corrs[_i]] if corrs[_i]!=np.nan else 0 for _i in range(9)]
        # data = [[f'X{_i + 1}_y', feat_set, corrs[_i]] for _i in range(9)]
        HEADER = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR', '1st-TTL']
        data = [[f'({HEADER[_i]}, y)', feat_set, corrs[_i]] for _i in range(9)]

        fig = plt.figure(figsize=(10, 5))  # (width, height)

        df = pd.DataFrame(data, columns=[f'Xi_y', 'feat_set', 'corr_rho'])
        g = sns.barplot(x=f"Xi_y", y="corr_rho", hue='feat_set', data=df)  # palette=palette,
        g.set(xlabel=None)
        g.set(ylabel='Rho')
        g.set_ylim(-1, 1)
        # g.set_title(dataset_name)
        g.get_legend().set_visible(True)  # False
        g.set_xticklabels(g.get_xticklabels(), fontsize=12, rotation=30, ha="center")

        ys = []
        xs = []
        width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            ys.append(height)
            xs.append(p.get_x())
            if i_p == 0:
                pre = p.get_x() + p.get_width()
            if i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width()
        plt.legend(loc='upper right')
        # # get the legend and modify it
        # handles, labels = g.get_legend_handles_labels()
        # fig.legend(handles, labels, title=None, loc='upper right', ncol=1,
        #  prop={'size': 8})  # loc='lower right',  loc = (0.74, 0.13)

        plt.tight_layout()

        title = dataset
        g.set_title(title + ' (header:True)', fontsize=13)
        out_file = os.path.join(out_dir, dataset, feat_set, "header:True", '-corr-bar.png')
        print(out_file)
        plt.savefig(out_file)  # should use before plt.show()
        if show: plt.show()
        plt.close(fig)
        plt.close("all")


def main():
    direction = 'src'
    if direction == 'src':
        in_dir = 'data/reprst_srcip'
        out_dir = 'out/reprst_srcip/correlation'
    else:  # src_dst
        in_dir = 'data/reprst'
        out_dir = 'out/reprst/correlation'

    print(os.path.abspath(os.getcwd()))
    corr_results = get_correlation(in_dir, out_dir, header=True)

    plot_correlation(corr_results, out_dir, show=True)
    plot_correlation_multi(corr_results, out_dir, show=True)


if __name__ == '__main__':
    main()
