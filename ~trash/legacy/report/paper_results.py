""" Extract AUCs from "xxx.xlxs" and plot the AUCs.

Run:
    root_dir="examples/reprst"
    python3.7 -u ${root_dir}/report/paper_results.py > ${root_dir}/out/report.txt 2>&1

"""

import os
import sys
import textwrap
import traceback
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

######################################################################################################################
### set seaborn background colors
# sns.set_style("darkgrid")
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

    if feat_type == "basic_representation".upper():
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

    elif feat_type == "effect_size".upper():
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
    elif feat_type == "effect_header".upper():
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


def parse_xlsx(in_file='xxx.xlsx'):
    """ Parse xlsx file

    Parameters
    ----------
    in_file

    Returns
    -------
        values_lst: list
            parsed resutls
    """
    xls = pd.ExcelFile(in_file)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)
    values_dict = OrderedDict()
    for i, sheet_name in enumerate(xls.sheet_names):
        print(i, sheet_name)
        if sheet_name.upper() not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
            continue
        # index_col=False: not use the first columns as index
        df = pd.read_excel(in_file, sheet_name=sheet_name, header=0, index_col=None)

        values_dict[sheet_name] = []
        for j, line in enumerate(list(df.values)):
            values_dict[sheet_name].append(line)

    return values_dict


def process_samp_value(samp_value, value_type='best+min'):
    """Parse SAMP-based features.

    Parameters
    ----------
    samp_value
    value_type

    Returns
    -------

    """
    try:
        v_arr = samp_value.split('|')
    except Exception as e:
        print(samp_value, value_type, e)
        return '0'
    if value_type == 'best+min':
        # v = v_arr[0] + "$|$"  # auc(q_samp|min)
        v_min = min([float(_v.replace(" ", '')) for _v in v_arr[4].split('=')[-1].split('+')])
        v_t = v_arr[0].split("(")  # best_auc(q_samp
        v_best = v_t[0]
        q_t = v_t[1].split("=")[1]  # q_samp=0.3
        # each cell: (best - min) qs=0.2
        v = "(" + f"{float(v_best):.2f} - " + f"{v_min:.2f}) " + "$q_{s}$=" + f"{float(q_t):.2f}"
    elif value_type == 'best+min+u' or value_type == 'best+min+avg':
        v = v_arr[0].split('(')[0] + "("  # auc(min|avg)
        v_min = min([float(_v.replace(" ", "")) for _v in v_arr[4].split('=')[-1].split('+')])
        # v += f"{v_min:.4f}$|$" + v_arr[2].split('=')[-1] + ")"
        # each cell:
        v_t = v_arr[0].split("(")  # best_auc(q_samp
        v_best = v_t[0]
        v_avg = v_arr[2].split('=')[-1]
        v = f"({float(v_best):.2f} - " + f"{v_min:.2f} - " + f"{float(v_avg):.2f})"
    else:
        print(f"value_type: {value_type} isn't implemented.")
    return v.replace('q_samp', '$q_{s}$')


def process_value(v):
    """Process each cell value

    Parameters
    ----------
    v

    Returns
    -------

    """
    v_t = v.split("(")[0]
    v_t = f"{float(v_t):.2f}"
    return v_t


def parse_data(value_lst, gs, datasets, res_type):
    """ parse the xlsx data

    Parameters
    ----------
    value_lst: xlsx data
    gs: boolean list
        True: get the results with best parameters
        False: get the results with default parameters
    datasets:
        which datasets we want
    res_type
        which case
    Returns
    -------
        needed_results: dict

    """
    needed_results = {}
    fig_flg = True
    if gs:
        value_lst = value_lst[15:]
    else:
        value_lst = value_lst[:15]

    for j, v in enumerate(value_lst):
        # if "sf:True-q_flow" not in str(v[0]):  # find the first line of value
        #     continue
        exist_data_flg = True
        if ("UNB/CICIDS_2017/pc_192.168.10.5" in str(v[1]) and 'UNB(PC1)' in datasets):
            data_name = 'UNB(PC1)'
        elif ("UNB/CICIDS_2017/pc_192.168.10.14" in str(v[1]) and 'UNB(PC4)' in datasets):
            data_name = 'UNB(PC4)'
        elif ('CTU/IOT_2017/pc_10.0.2.15' in str(v[1]) and 'CTU' in datasets):
            data_name = 'CTU'
        elif ('MAWI/WIDE_2019/pc_202.171.168.50' in str(v[1]) and 'MAWI' in datasets):
            data_name = 'MAWI'
        elif ('UCHI/IOT_2019/smtv_10.42.0.1' in str(v[1]) and 'TV&RT' in datasets):
            data_name = 'TV&RT'
        elif ('UCHI/IOT_2019/sfrig_192.168.143.43' in str(v[1]) and 'SFrig' in datasets):
            data_name = 'SFrig'
        elif ('UCHI/IOT_2019/bstch_192.168.143.48' in str(v[1]) and 'BSTch' in datasets):
            data_name = 'BSTch'
        elif ("UNB/CICIDS_2017/pc_192.168.10.8" in str(v[1]) and 'UNB(PC2)' in datasets):
            data_name = 'UNB(PC2)'
        elif ("UNB/CICIDS_2017/pc_192.168.10.9" in str(v[1]) and 'UNB(PC3)' in datasets):
            data_name = 'UNB(PC3)'
        elif ("UNB/CICIDS_2017/pc_192.168.10.15" in str(v[1]) and 'UNB(PC5)' in datasets):
            data_name = 'UNB(PC5)'
        elif ('UCHI/IOT_2019/ghome_192.168.143.20' in str(v[1]) and 'GHom' in datasets):
            data_name = 'GHom'
        elif ('UCHI/IOT_2019/scam_192.168.143.42' in str(v[1]) and 'SCam' in datasets):
            data_name = 'SCam'
        else:
            # raise  ValueError()
            # exist_data_flg = False
            continue

        # get test size
        test_size = [t.split(':')[1] for t in str(v[3]).split()]

        num_feats = 13
        if exist_data_flg:
            if res_type.upper() == 'basic_representation'.upper():
                # representations = ["STATS", "IAT", "IAT-FFT", "SAMP-NUM", "SAMP-NUM-FFT", "SAMP-SIZE",
                #                    "SAMP-SIZE-FFT"]  # all without header
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[4]), process_value(v[7]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[14], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[15], value_type=value_type)
                         ]
                if exist_data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    needed_results[data_name] = (test_size, v_arr)

            elif res_type.upper() == 'effect_size'.upper():
                # representations = ["STATS", "SIZE", "IAT", "IAT+SIZE", "SAMP-NUM", "SAMP-SIZE"]  # all without header
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[5]), process_value(v[4]), process_value(v[6]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type)]
                if exist_data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    needed_results[data_name] = (test_size, v_arr)

            elif res_type.upper() == 'effect_header'.upper():
                # representations = ["STATS (wo. header)", "STATS (w. header)", "IAT+SIZE (wo. header)",
                #                    "IAT+SIZE (w. header)", "SAMP-SIZE (wo. header)",
                #                    "SAMP-SIZE (w. header)"]
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[10 + num_feats + 1]),
                         process_value(v[6]), process_value(v[6 + num_feats + 1]),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type),
                         ]
                if exist_data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    needed_results[data_name] = (test_size, v_arr)

            else:
                # v = change_line(v_arr, k=2, value_type=value_type)
                pass
        else:
            pass

    return needed_results


def merge_xlsx(in_files=[], out_file='merged.xlsx'):
    """ merge multi-xlsx files to one.

    Parameters
    ----------
    in_files: list of 2-element tuples (name, file_path)
    out_file: merged xlsx

    Returns
    -------
        None
    """

    workbook = xlsxwriter.Workbook(out_file)
    for t, (sheet_name, in_file) in enumerate(in_files):
        # print(t, sheet_name, in_file)
        worksheet = workbook.add_worksheet(sheet_name)
        if not os.path.exists(in_file):
            # for i in range(rows):
            #     worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
            #                                    list(values[i])])  # write a list from (row, col)
            raise FileNotFoundError(in_file)
        else:
            df = pd.read_excel(in_file, header=None, index_col=None)
            values = df.values
            rows, cols = values.shape
            ### add column index
            # worksheet.write_row(0, 0, [str(i) for i in range(cols)])
            for i in range(rows):
                worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
                                           list(values[i])])  # write a list from (row, col)
            # worksheet.write(df.values)
    workbook.close()


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


def get_part_results(needed_results, res_type, r_th, cols):
    """ only get the r-th row results.

    Parameters
    ----------
    needed_results: dict
    res_type
    r_th: int

    cols: int
        the number of columns of the figure

    Returns
    -------

    """
    new_data = OrderedDict()  # not used yet, for future purposes.

    part_results = []
    y_range = []  # get all y values and use them to decide the ticks of y-axis.
    for j, (detector_name, vs_dict) in enumerate(needed_results.items()):
        if j >= r_th * cols and j < (r_th + 1) * cols:
            sub_dataset = []
            new_colors = []
            yerrs = []
            for ind, (dataset_name, (test_size, vs)) in enumerate(vs_dict.items()):
                test_size = sum([int(v) for v in test_size])
                yerr = []
                if dataset_name not in new_data.keys():
                    new_data[dataset_name] = {detector_name: {}, 'test_size': [], 'yerr': []}

                if res_type.upper() == "basic_representation".upper():
                    yerr = [1 / np.sqrt(test_size)] * 3  # for error bar
                    _std = 1 / np.sqrt(test_size)

                    features = ['STATS', 'IAT', 'IAT-FFT', 'SAMP-NUM', 'SAMP-NUM-FFT', 'SAMP-SIZE', 'SAMP-SIZE-FFT']
                    f_dict = dict(zip(features, [i for i in range(len(features))]))
                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'IAT'
                    B = 'IAT-FFT'
                    pre_aucs = [float(vs[f_dict[A]]), 0]
                    aucs = [float(vs[f_dict[B]]), 0]
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ {A}"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name] = {repres_pair: diff[0]}
                    print(dataset_name, detector_name, 'IAT:', pre_aucs, 'IAT-FFT:', aucs)
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'SAMP-NUM'
                    B = 'SAMP-NUM-FFT'
                    max_auc, min_auc = vs[f_dict[A]].split('(')[1].split(')')[0].split('-')
                    pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    max_auc, min_auc = vs[f_dict[B]].split('(')[1].split(')')[
                        0].split('-')
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ {A}"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name][repres_pair] = diff[0]
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'SAMP-SIZE'
                    B = 'SAMP-SIZE-FFT'
                    max_auc, min_auc = vs[f_dict[A]].split('(')[1].split(')')[
                        0].split('-')
                    pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    max_auc, min_auc = vs[f_dict[B]].split('(')[1].split(')')[
                        0].split('-')
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ {A}"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name][repres_pair] = diff[0]
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    # get the colors for each bar
                    new_colors = seaborn_palette(res_type, fig_type='diff').values()

                elif res_type.upper() == "effect_size".upper():
                    yerr = [1 / np.sqrt(test_size)] * 2
                    _std = 1 / np.sqrt(test_size)

                    features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP-SIZE']
                    f_dict = dict(zip(features, [i for i in range(len(features))]))

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'IAT'
                    B = 'IAT+SIZE'
                    pre_aucs = [float(vs[f_dict[A]]), 0]
                    aucs = [float(vs[f_dict[B]]), 0]
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ {A}"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name] = {repres_pair: diff[0]}
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'SAMP-NUM'
                    B = 'SAMP-SIZE'
                    max_auc, min_auc = vs[f_dict[A]].split('(')[1].split(')')[
                        0].split('-')
                    pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    max_auc, min_auc = vs[f_dict[B]].split('(')[1].split(')')[
                        0].split('-')
                    aucs = [float(max_auc), float(min_auc)]
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ {A}"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name][repres_pair] = diff[0]
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    new_colors = seaborn_palette(res_type, fig_type='diff').values()

                elif res_type.upper() == "effect_header".upper():
                    yerr = [1 / np.sqrt(test_size)] * 3
                    _std = 1 / np.sqrt(test_size)

                    features = ['STATS (wo. header)', 'STATS (w. header)', 'IAT+SIZE (wo. header)',
                                'IAT+SIZE (w. header)',
                                'SAMP-SIZE (wo. header)', 'SAMP-SIZE (w. header)']
                    f_dict = dict(zip(features, [i for i in range(len(features))]))

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'STATS (wo. header)'
                    B = 'STATS (w. header)'
                    pre_aucs = [float(vs[f_dict[A]]), 0]
                    aucs = [float(vs[f_dict[B]]), 0]
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ (wo. header)"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name] = {repres_pair: diff[0]}
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    A = 'IAT+SIZE (wo. header)'
                    B = 'IAT+SIZE (w. header)'
                    pre_aucs = [float(vs[f_dict[A]]), 0]
                    aucs = [float(vs[f_dict[B]]), 0]
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ (wo. header)"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name][repres_pair] = diff[0]
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                    A = 'SAMP-SIZE (wo. header)'
                    B = 'SAMP-SIZE (w. header)'
                    max_auc, min_auc = vs[f_dict[A]].split('(')[1].split(')')[
                        0].split('-')
                    pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    max_auc, min_auc = vs[f_dict[B]].split('(')[1].split(')')[
                        0].split('-')
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                    diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                    repres_pair = f"{B} \\ (wo. header)"
                    sub_dataset.append((dataset_name, repres_pair, diff[0]))
                    new_data[dataset_name][detector_name][repres_pair] = diff[0]
                    y_range.append(diff[0] + _std if diff[0] > 0 else diff[0] - _std)

                    new_colors = seaborn_palette(res_type, fig_type='diff').values()

                new_data[dataset_name]['test_size'] = test_size
                new_data[dataset_name]['yerr'] = yerr

                yerrs.append(yerr)
            part_results.append([detector_name, sub_dataset, new_colors, yerrs])

    return part_results, y_range


def plot_bar_difference_seaborn(needed_results={}, methods=['OCSVM', 'AE'],
                                datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'],
                                fig_params={'gs': True, 'res_type': ''},
                                out_file="F1_for_all.pdf"):
    """ process the needed_results,  get the AUC differences, and plot the diffence.

    Parameters
    ----------
    needed_results: dict
        needed to further process
    methods:
        which methods will be shown in the figure
    datasets:
        which datasets will be shown
    out_file:
        save the figure to disk
    fig_params: parameters for plotting the resutls

    Returns
    -------

    """
    #############################################################################################################
    ### create plots
    n_figs = len(methods)
    if len(datasets) <= 7:  # we display the results in 2 columns.
        # cols of subplots in each row
        cols = 2
        if n_figs > cols:
            if n_figs % cols == 0:
                rows = int(n_figs // cols)
            else:
                rows = int(n_figs // cols) + 1
            fig, axes = plt.subplots(rows, cols, figsize=(18, 10))  # (width, height)
            # axes = axes.reshape(rows, -1)\
    else:  # we display the results in a single column.
        cols = 1  # cols of subplots in each row
        rows = n_figs
        fig, axes = plt.subplots(rows, cols, figsize=(18, 25))  # (width, height)
        axes = axes.reshape(rows, -1)
    print(f'subplots: ({rows}, {cols})')

    #############################################################################################################
    ### For each row of the figure, we get the partial results.
    ### e.g., when cols = 2, for each row, we get the results generated by two methods.
    res_type = fig_params['res_type']
    for r_th in range(rows):
        ### get each row results, i.e., compute all AUC differences. here 'ys' is used to decide the y-axis range.
        part_results, ys = get_part_results(needed_results, res_type, r_th, cols)
        _y_min, _y_max, _ys = get_ylim(ys)

        for c_th, (detector_name, _single_result, _new_colors, _yerrs) in enumerate(part_results):
            df = pd.DataFrame(_single_result, columns=['dataset', 'repres', 'diff'])

            # For each AUC difference, we get the corresponding error bar.
            new_yerrs = []
            _yerrs = np.asarray(_yerrs)
            for c_tmp in range(_yerrs.shape[1]):  # extend by columns
                new_yerrs.extend(_yerrs[:, c_tmp])
            # print('new_yerrs:', new_yerrs)

            # plot the (r_th, c_th) subfigure.
            g = sns.barplot(y="diff", x='dataset', hue='repres', data=df, palette=_new_colors, ci=None,
                            capsize=.2, ax=axes[r_th, c_th % cols])

            # compute the error bar position
            ys = []
            xs = []
            width = 0
            for i_p, p in enumerate(g.patches):
                height = p.get_height()
                width = p.get_width()
                ys.append(height)
                xs.append(p.get_x())
                num_bars = df['repres'].nunique()

                if i_p == 0:
                    pre = p.get_x() + p.get_width() * num_bars
                    # sub_fig_width = p.get_bbox().width
                if i_p < df['dataset'].nunique() and i_p > 0:
                    cur = p.get_x()
                    g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                    pre = cur + p.get_width() * num_bars

            axes[r_th, c_th % cols].errorbar(x=xs + width / 2, y=ys, yerr=new_yerrs, fmt='none', cols='b', capsize=3)
            g.set(xlabel=None)
            g.set(ylabel=None)

            font_size = 20
            g.set_ylabel('AUC difference', fontsize=font_size + 4)
            # only the last row shows the xlabel.
            if rows > 1:
                if r_th < rows - 1:
                    g.set_xticklabels([])
                else:
                    g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")

            g.set_ylim(_y_min, _y_max)
            # set the yticks and labels
            g.set_yticks(_ys)
            _ys_fmt = get_yticklabels(_ys)
            g.set_yticklabels(_ys_fmt, fontsize=font_size + 6)
            print(g.get_yticks(), _ys)

            ### don't show the y ticks and labels for rest of columns.
            if c_th % cols != 0:
                # g.get_yaxis().set_visible(False)
                g.set_yticklabels(['' for v_tmp in _ys])
                g.set_ylabel('')

            # set title for each subplot
            g.set_title(detector_name, fontsize=font_size + 8)
            g.get_legend().set_visible(False)

    ### get the legend from the last 'ax' (here is 'g') and relocated its position.
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    fig.legend(handles, labels, loc='lower center', ncol=3, prop={'size': font_size - 2}, frameon=False)

    plt.tight_layout()
    if cols == 2:
        plt.subplots_adjust(bottom=0.17)
    else:
        plt.subplots_adjust(bottom=0.07)

    plt.savefig(out_file)  # should use before plt.show()
    # plt.show()
    plt.close(fig)


def dat2tables(data, out_file='.txt'):
    """ data to latex tables

    Parameters
    ----------
    data
    out_file

    Returns
    -------

    """
    features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP-SIZE']
    keep_features = ['STATS', 'SIZE', 'IAT', 'SAMP-NUM']
    outs = {}
    for j, (detector_name, vs_dict) in enumerate(data.items()):
        sub_dataset = []
        for ind, (dataset_name, (test_size, vs)) in enumerate(vs_dict.items()):
            # test_size = int(test_size[0]) + int(test_size[1])    # test_size=('100', '100')
            f_dict = dict(zip(features, [i for i in range(len(features))]))
            tmp = [dataset_name]
            for feat in keep_features:
                if feat in ['SAMP-NUM', 'SAMP-SIZE']:
                    ### '(0.80 - 0.18) $q_{s}$=0.50'
                    max_auc, min_auc = vs[f_dict[feat]].split('(')[1].split(')')[0].split('-')
                    v = max_auc
                else:
                    v = vs[f_dict[feat]]
                tmp.append(v)
            sub_dataset.append(tmp)

        print(f'{detector_name}: {sub_dataset}')
        outs[detector_name] = sub_dataset

    # out to txt
    with open(out_file, 'w') as f:
        line = 'Detector & Dataset & ' + ' & '.join(keep_features) + '\n\\toprule\n'
        f.write(line)
        for detector_name, vs_lst in outs.items():
            # _n = len(dataset_name)
            line = "\multirow{12}{*}{~\\rule{0pt}{2.7ex}" + f'{detector_name}' + '}'
            for vs in vs_lst:
                line += ' & ' + ' & '.join(vs) + '\\\\ \n \\cmidrule{2-6}' + '\n'

            line += '\midrule' + '\n'
            f.write(line)
        f.write('\\bottomrule\n')


def xlsx2tables(results, name, gs, datasets, methods, res_type='effect_size', out_file=''):
    """

    Parameters
    ----------
    results
    name
    gs
    datasets
    methods
    res_type
    out_file

    Returns
    -------

    """
    with open(out_file, 'w') as out_f:
        needed_results = OrderedDict()
        for detector_name in methods:
            for i, (sheet_name, value_lst) in enumerate(results.items()):
                if detector_name == sheet_name:
                    print(detector_name)
                    data = parse_data(value_lst, gs, datasets, res_type)
                    needed_results[detector_name] = data

        ### get tables
        out_file = os.path.dirname(out_file) + f"/{name}.txt"
        print(out_file)
        dat2tables(needed_results, out_file)

    return out_file


def xlsx2figs(results, name, gs, datasets, methods, res_type='basic_representation', out_file=''):
    """ process the parsed results (such as, get the difference) and plot the results

    Parameters
    ----------
    results: dict
        the results extracted from xlsx
    name: str
        used for labeling the output file
    gs: boolean
        which results you want to get from the xlsx, best or default
    datasets: list
        which datasets
    methods: list
        which methods
    res_type: str
        which experimental case
    out_file: str
        output file

    Returns
    -------

    """
    with open(out_file, 'w') as out_f:
        # only extract the needed results from xlsx
        needed_results = OrderedDict()
        for detector_name in methods:
            for i, (sheet_name, value_lst) in enumerate(results.items()):
                if detector_name == sheet_name:
                    print(detector_name)
                    data = parse_data(value_lst, gs, datasets, res_type)
                    needed_results[detector_name] = data

        ### generate figure
        out_file = os.path.dirname(out_file) + f"/{name}.pdf"
        print(out_file)
        plot_bar_difference_seaborn(needed_results, methods, datasets,
                                    fig_params={'gs': gs, 'res_type': res_type},
                                    out_file=out_file)
    return out_file


def _main(in_dir, results, gses=[True, False],
          datasets_lst=[['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']],
          methods_lst=[['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]):
    """ For each case (i.e, 'basic_representation', 'effect_size', 'effect_header'), we plot the result.
        'basic_representation': FFT vs. no FFT
        'effect_size': SIZE results
        'effect_header': Header results

    Parameters
    ----------
    in_dir: dir
    results: dict
        parsed results from the merged xlsx
    gses: boolean list
        True means we want to get the results with best parameters.
        False means we want to get the results with default parameters.
    datasets_lst: list
        Which datasets' results we wanted
    methods_lst
        Which methods's resutls we wanted
    Returns
    -------

    """
    for gs in gses:
        for datasets in datasets_lst:
            for methods in methods_lst:
                for res_type in ['basic_representation', 'effect_size', 'effect_header']:
                    tmp = 'best' if gs else 'default'
                    name = '_'.join(methods) + f'-{len(datasets)}-' + tmp + '-' + res_type
                    out_file = f'{in_dir}/res/{name}-latex_tables_figs.txt'
                    check_path(out_file)
                    print(f'\n\n******************\n{out_file}')
                    try:
                        # xlsx2tables(results, name, gs, datasets, methods, 'effect_size', out_file)
                        xlsx2figs(results, name, gs, datasets, methods, res_type, out_file)
                    except Exception as e:
                        print('Error: ', e)
                        traceback.print_exc()
                    # break


def main(root_dir='examples/reprst'):
    """Get results from xlsx and plot the results.

    Parameters
    ----------
    root_dir

    Returns
    -------

    """
    # in_dir = f'{root_dir}/out/report/reprst_srcip/20201227'
    in_dir = f'auc_results'
    ########################################################################################################
    ### merge all results to one xlsx
    in_files = [("OCSVM", f'{in_dir}/OCSVM.txt.dat.csv.xlsx_highlight.xlsx'),
                ("KDE", f'{in_dir}/KDE.txt.dat.csv.xlsx_highlight.xlsx'),
                ("GMM", f'{in_dir}/GMM.txt.dat.csv.xlsx_highlight.xlsx'),
                ("IF", f'{in_dir}/IF.txt.dat.csv.xlsx_highlight.xlsx'),
                ("PCA", f'{in_dir}/PCA.txt.dat.csv.xlsx_highlight.xlsx'),
                ("AE", f'{in_dir}/AE.txt.dat.csv.xlsx_highlight.xlsx')
                ]
    xlsx_file = f'{in_dir}/Results-merged.xlsx'
    print('merge xlsx.')
    merge_xlsx(in_files, xlsx_file)

    ########################################################################################################
    ### Get the results on part of datasets and all algorithms
    # 1. datasets: [UNB(PC1), UNB(PC4), CTU, MAWI, TV&RT, SFrig, and BSTch]
    # algorithms: [OCSVM, IF, AE, KDE, GMM, PCA]
    ### parse the results from the xlsx
    print('parse xlsx.')
    results = parse_xlsx(xlsx_file)
    gses = [True, False]  # if get the best results from the the merged xlsx or not.
    datasets_lst = [['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']]
    methods_lst = [['OCSVM', 'IF', 'AE', 'KDE', 'GMM', 'PCA']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)

    ########################################################################################################
    ### Get the results of the rest of datasets.
    # 2. datasets: ['UNB(PC2)', 'UNB(PC3)',  'UNB(PC5)', 'GHom', 'SCam']
    print('parse xlsx.')  # reparse xlsx to avoid any changes that may happen in previous operations.
    results = parse_xlsx(xlsx_file)
    datasets_lst = [['UNB(PC2)', 'UNB(PC3)', 'UNB(PC5)', 'GHom', 'SCam']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)

    ########################################################################################################
    ### 3. Get the results on all datasets and algorithms
    print('parse xlsx.')
    results = parse_xlsx(xlsx_file)  # reparse xlsx to avoid any changes that may happen in previous operations.
    gses = [True, False]
    datasets_lst = [['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)', 'CTU', 'MAWI', 'TV&RT', 'GHom',
                     'SCam', 'SFrig', 'BSTch']]
    methods_lst = [['OCSVM', 'IF', 'AE', 'KDE', 'GMM', 'PCA']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)


if __name__ == '__main__':
    main()
