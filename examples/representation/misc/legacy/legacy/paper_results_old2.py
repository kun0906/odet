"""Data postprocess including data format transformation, highlight data

Run:
    root_dir="examples/reprst" | python3.7 -u ${root_dir}/report/paper_results.py > ${root_dir}/out/report.txt 2>&1

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import os
import sys
import traceback

lib_path = os.path.abspath('.')
sys.path.append(lib_path)
# print(f"add \'{lib_path}\' into sys.path: {sys.path}")
#
# import matplotlib
# matplotlib.use('TkAgg')     # pycharm can't use subplots_adjust() to show the legend and modified subplots correctly.

import copy
from operator import itemgetter

import xlsxwriter

from itod.utils.tool import *
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import ListedColormap

import seaborn as sns

# sns.set_style("darkgrid")

# colorblind for diff
sns.set_palette("bright")  # for feature+header
sns.palplot(sns.color_palette())


def split_str(v_str, value_type='best+min'):
    if value_type == 'best+min':
        _v_t = v_str.strip().split(')')
        q_v = _v_t[-1]
        _v_t_1 = _v_t[0].split('(')[1].split('-')
        best_v = _v_t_1[0]
        min_v = _v_t_1[1]
        return best_v, min_v, q_v
    elif value_type == 'best+min+avg':
        _v_t = v_str.strip().split(')')[0].split('-')
        best_v = _v_t[0].split('(')[1]
        min_v = _v_t[1]
        avg_v = _v_t[-1]

        return best_v, min_v, avg_v


def find_top_k_values(arr, k=2, reverse=False):
    # indexs = sorted(range(len(arr)), key=lambda k: arr[k], reverse=True)
    # L = [2, 3, 1, 4, 5]

    indices, arr_sorted = zip(*sorted(enumerate(arr), key=itemgetter(1), reverse=reverse))
    same_values = [0] * len(indices)
    c = 0
    for i, (idx, v) in enumerate(zip(indices, arr_sorted)):
        if i == 0:
            same_values[i] = 0  # color index
            continue
        if v == arr_sorted[i - 1]:  # for the same value
            k += 1
            # same_values[i] = c
        else:
            c += 1
        same_values[i] = c

        if i + 1 > k:
            break

    return arr_sorted[0:k], indices[0:k], same_values[0:k]


def split_str(v_str, value_type='best+min'):
    if value_type == 'best+min':
        _v_t = v_str.strip().split(')')
        q_v = _v_t[-1]
        _v_t_1 = _v_t[0].split('(')[1].split('-')
        best_v = _v_t_1[0]
        min_v = _v_t_1[1]
        return best_v, min_v, q_v
    elif value_type == 'best+min+avg':
        _v_t = v_str.strip().split(')')[0].split('-')
        best_v = _v_t[0].split('(')[1]
        min_v = _v_t[1]
        avg_v = _v_t[-1]

        return best_v, min_v, avg_v


def csv2latex(input_file='', tab_latex='', caption='', do_header=False, num_feat=4, verbose=1):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        gs_false = False
        gs_ture = False
        k = 2
        tab_latex_false = copy.deepcopy(tab_latex)
        tab_latex_true = copy.deepcopy(tab_latex)
        for i, line in enumerate(lines):
            arr = line.strip().split(',')
            if arr == [''] or 'Dataset, Train set, Test set, IAT, IAT_sizes' in line:
                continue
            if arr[0].startswith('gs:False'):
                if verbose:
                    print(f'save {arr[0]}')
                t = 0
                tab_latex_false[2] = tab_latex_false[2].replace('{v}', caption + '_default_parameters')
                tab_latex_false[3] = tab_latex_false[3].replace('{v}', caption + ': default parameters')
                gs_false = True
                gs_ture = False
                continue
            if gs_false:
                if do_header:
                    if num_feat == 3:
                        arr_tmp = [arr[11], arr[14], arr[15]]
                    else:  # num_feat ==4
                        arr_tmp = [arr[13], arr[14], arr[15], arr[16], arr[17], arr[18], arr[19]]
                else:
                    if num_feat == 3:
                        arr_tmp = [arr[4], arr[7], arr[8]]
                    else:
                        arr_tmp = [arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10]]

                arr_tmp = change_line(arr_tmp, k=k)
                # tab_latex_true[8 + t] = tab_latex_true[8+t].format(arr_tmp)
                for _ in range(len(tab_latex_false[(8 + t):-1])):
                    if '\midrule' in tab_latex_false[8 + t]:
                        t += 1
                    else:
                        break
                # tab_latex_false[8+t] = tab_latex_false[8+t].format(arr_tmp)
                tab_latex_false[8 + t] = tab_latex_false[8 + t].replace('{v}', arr_tmp)
                t += 1

            if arr[0].startswith('gs:True'):
                gs_false = False
                if verbose:
                    print('--- gs:False\n')
                # if previous_results:
                #     ### swap the results of SMTV and MAWI
                #     tab_latex_false[]
                print('\n'.join(tab_latex_false))

                if verbose:
                    print(f'save {arr[0]}')
                tab_latex_true[2] = tab_latex_true[2].replace('{v}', caption + '_best_parameters')
                tab_latex_true[3] = tab_latex_true[3].replace('{v}', caption + ': best parameters')
                gs_ture = True
                t = 0
                continue

            if gs_ture:
                if do_header:
                    if num_feat == 3:
                        arr_tmp = [arr[11], arr[14], arr[15]]
                    else:  # num_feat ==4
                        arr_tmp = [arr[13], arr[14], arr[15], arr[16], arr[17], arr[18], arr[19]]
                else:
                    if num_feat == 3:
                        arr_tmp = [arr[4], arr[7], arr[8]]
                    else:
                        arr_tmp = [arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10]]

                arr_tmp = change_line(arr_tmp, k=k)

                # tab_latex_true[8 + t] = tab_latex_true[8+t].format(arr_tmp)
                for _ in range(len(tab_latex_true[(8 + t):-1])):
                    if '\midrule' in tab_latex_true[8 + t]:
                        t += 1
                    else:
                        break
                tab_latex_true[8 + t] = tab_latex_true[8 + t].replace('{v}', arr_tmp)
                t += 1

            if verbose:
                print(f'{i}, {line.strip()}')
        if verbose:
            print('+++ gs:True\n')
        print('\n'.join(tab_latex_true))

        return tab_latex_false, tab_latex_true


def figure_temple(name='', caption='', cols_format='', label='', cols=[], output_file='.pdf'):
    if name == 'diff':
        figure_latex = [
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.99\\textwidth]{" + f"{output_file}" + "}",  # trim=0cm 0cm 0cm 0cm
            "\\caption{" + f"{caption}" + "}",
            "\\label{fig:" + f"{label}" + "}",
            "\\end{figure}"
        ]
    else:
        figure_latex = [
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.80\\textwidth]{" + f"{output_file}" + "}",
            "\\caption{" + f"{caption}" + "}",
            "\\label{fig:" + f"{label}" + "}",
            "\\end{figure}"
        ]
    return figure_latex


def comb_figures_temple(name='', detector_name='', caption='',
                        cols_format='', label='', cols=[], fig1='.pdf', fig1_label='',
                        fig2='.pdf', fig2_label='', fig_type='default'):
    figure_latex = [
        "\\begin{figure}[H]",
        "\\centering",
        "\\begin{subfigure}{.5\\textwidth}",
        "   \\centering",
        "   \\includegraphics[width=.99\\linewidth]{" + f"{fig1}" + "}",
        "   \\caption{Best parameters}",
        "   \\label{fig:" + f"{fig1_label}" + "}",
        "\\end{subfigure}%",
        "\\begin{subfigure}{.5\\textwidth}",
        "   \\centering",
        "   \\includegraphics[width=.99\\linewidth]{" + f"{fig2}" + "}",
        "   \\caption{Default parameters}",
        "   \\label{fig:" + f"{fig2_label}" + "}",
        "\\end{subfigure}",
        "\\caption{" + f"{caption}" + "}",
        "\\label{" + f"{label}" + "}",
        "\\end{figure}"
    ]
    return figure_latex


def seaborn_palette(feat_type='', fig_type='diff'):
    # 1 bright used for basic
    # Set the palette to the "pastel" default palette:
    sns.set_palette("bright")
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_bright = sns.color_palette()

    # muted for FFT
    sns.set_palette("muted")  # feature+size
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_muted = sns.color_palette()

    # dark for feature + size
    sns.set_palette("dark")
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_dark = sns.color_palette()

    # deep for feature + header
    sns.set_palette("deep")  # for feature+header
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_deep = sns.color_palette()

    # colorblind for diff
    sns.set_palette("colorblind")  # for feature+header
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_colorblind = sns.color_palette()

    # construct cmap
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    # my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    colors_bright = ListedColormap(colors_bright.as_hex()).colors
    colors_dark = ListedColormap(colors_dark.as_hex()).colors
    colors_muted = ListedColormap(colors_muted.as_hex()).colors
    colors_deep = ListedColormap(colors_deep.as_hex()).colors
    colors_colorblind = ListedColormap(colors_colorblind.as_hex()).colors
    # colors_bright = ListedColormap(colors_bright.as_hex())

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


def parse_xlsx(input_file='xxx.xlsx'):
    xls = pd.ExcelFile(input_file)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)
    values_dict = OrderedDict()
    for i, sheet_name in enumerate(xls.sheet_names):
        print(i, sheet_name)
        if sheet_name.upper() not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
            continue
        df = pd.read_excel(input_file, sheet_name=sheet_name, header=0,
                           index_col=None)  # index_col=False: not use the first columns as index

        values_dict[sheet_name] = []
        for j, line in enumerate(list(df.values)):
            values_dict[sheet_name].append(line)

    return values_dict


def process_samp_value(samp_value, value_type='best+min'):
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
    v_t = v.split("(")[0]
    v_t = f"{float(v_t):.2f}"
    return v_t


def change_line(arr_tmp, k=2, value_type='best+min'):
    arr_tmp = copy.copy(arr_tmp)

    def color_value(v, format='bold'):
        if format == 'bold':
            v_str = "\\textbf{" + v.strip() + "}"
        elif format == 'dark_grey':
            v_str = "\\textcolor[rgb]{0.300,0.300,0.300}{" + v.strip() + "}"
        elif format == 'weak_grey':
            v_str = "\\textcolor[rgb]{0.400,0.400,0.400}{" + v.strip() + "}"

        return v_str

    def ordered_color(v, k=0, value_type='best+min'):

        if k == 0:
            if ')' in v:
                if value_type == 'best+min':
                    best_v, min_v, q_v = split_str(v, value_type=value_type)
                    v_ind = '(' + color_value(best_v, format='bold') + ' - ' + str(min_v) + ') ' + q_v
                elif value_type == 'best+min+avg':
                    best_v, min_v, avg_v = split_str(v, value_type=value_type)
                    v_ind = '(' + color_value(best_v, format='bold') + ' - ' + str(min_v) + ' - ' + avg_v + ')'
            else:
                v_ind = color_value(v, format='bold')
        elif k == 1:
            if ')' in v:
                if value_type == 'best+min':
                    best_v, min_v, q_v = split_str(v, value_type=value_type)
                    v_ind = '(' + color_value(color_value(best_v, format='dark_grey'), format='bold') + ' - ' + str(
                        min_v) + ') ' + q_v
                elif value_type == 'best+min+avg':
                    best_v, min_v, avg_v = split_str(v, value_type=value_type)
                    v_ind = '(' + color_value(color_value(best_v, format='dark_grey'), format='bold') + ' - ' + str(
                        min_v) + ' - ' + avg_v + ')'

            else:
                v_ind = color_value(color_value(v, format="dark_grey"), format='bold')
        return v_ind

    try:
        # arr_tmp = [float(v) for v in arr_tmp]
        arr_tmp_f = []
        for v in arr_tmp:
            if '(' in v:
                # cell format: best(q_s=0.3|min)
                # v = v.split('(')[0]
                # cell format: (best-min)q_s=0.3
                # print(v.split('-'), arr_tmp)
                v = v.split('-')[0].split("(")[1]
            arr_tmp_f.append(float(v))
        arr_sorted, indices, same_values = find_top_k_values(arr_tmp_f, k=k, reverse=True)
        for j, (ind, s_v) in enumerate(zip(indices, same_values)):  #
            v = arr_tmp[ind]
            v_ind = ordered_color(v, k=s_v, value_type=value_type)
            arr_tmp[ind] = str(v_ind)  # updated arr_tmp
    except Exception as e:
        print(f"Error: {e}, {arr_tmp}")

    for i, v in enumerate(arr_tmp):
        try:
            if 'q_' in v:
                arr_tmp[i] = v.replace('$q_{s}$', "\\textcolor[rgb]{0.400,0.400,0.400}{$q_{s}$") + "}"
        except:
            print(i, arr_tmp)
            continue
    arr_tmp = ' & '.join([str(v) for v in arr_tmp])

    return arr_tmp


def get_auc_from_arr(arr, repres='IAT'):
    # step=4  # the beginning of representation without header
    # h_step = 18 # the beginning of representation with header
    # if gs:
    #     if header:
    #         line = arr[h_step:]
    #     else:
    #         line = arr[step:h_step-1]
    # else:
    #     if header:
    #         line = arr[h_step:]
    #     else:
    #         line = arr[step:h_step-1]

    all_repres = ['IAT', 'SIZE', 'IAT+SIZE', 'FFT_IAT', 'FFT_SIZE', 'FFT_IAT+SIZE', 'STAT', 'SAMP_NUM',
                  'SAMP_SIZE', 'SAMP_NUM+SIZE', 'FFT_SAMP_NUM', 'FFT_SAMP_SIZE', 'FFT_SAMP_NUM+SIZE']
    for i, repres_i in enumerate(all_repres):
        if repres_i == repres:
            auc = float(arr[0].strip())
            break

    if 'SAMP-' in repres.upper():
        max_auc, min_auc = process_samp_value(auc, value_type='best+min')
        aucs = [float(max_auc), float(min_auc)]  # max and min aucs
    else:
        aucs = [float(process_value(auc)), 0]

    return aucs


def get_dimension_from_cell(v_cell, value_type=''):
    # 0.8888(q=0.9|dim=14)
    dim = v_cell.split('dim=')[1]
    if "SAMP".upper() in v_cell.upper():
        dim = dim.split('|')[0]
    else:
        dim = dim.split(')')[0]

    return dim


def parse_data(value_lst, gs, datasets, detector_name, res_type, out_file=''):
    data = {}
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
            exist_data_flg = False
            continue

        test_size = [t.split(':')[1] for t in str(v[3]).split()]

        num_feats = 13
        if exist_data_flg:
            if res_type.upper() == 'basic_representation'.upper():
                representations = ["STATS", "IAT", "IAT-FFT", "SAMP-NUM", "SAMP-NUM-FFT", "SAMP-SIZE",
                                   "SAMP-SIZE-FFT"]  # all without header
                # colors = ['m', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(res_type, fig_type='raw')
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
                    data[data_name] = (test_size, v_arr)
                v = change_line(v_arr, k=2, value_type=value_type)

            elif res_type.upper() == 'effect_size'.upper():
                representations = ["STATS", "SIZE", "IAT", "IAT+SIZE", "SAMP-NUM", "SAMP-SIZE"]  # all without header
                # colors = ['m', 'blue', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(res_type, fig_type='raw')
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
                    data[data_name] = (test_size, v_arr)
                v = change_line(v_arr, k=2, value_type=value_type)  # it will change v_arr values

            elif res_type.upper() == 'effect_header'.upper():
                representations = ["STATS (wo. header)", "STATS (w. header)", "IAT+SIZE (wo. header)",
                                   "IAT+SIZE (w. header)", "SAMP-SIZE (wo. header)",
                                   "SAMP-SIZE (w. header)"]
                # colors = ['m', 'purple', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(res_type, fig_type='raw')
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
                    data[data_name] = (test_size, v_arr)
                v = change_line(v_arr, k=2, value_type=value_type)

            # elif res_type.upper() == 'appd_all_without_header'.upper():
            #     representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
            #     # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
            #     colors = seaborn_palette(res_type, fig_type='raw')
            #     value_type = 'best+min'
            #     v_arr = [process_value(v[4]), process_value(v[7]), process_value(v[5]), process_value(v[8]),
            #              process_samp_value(v[11], value_type=value_type),
            #              process_samp_value(v[12], value_type=value_type)]
            #     if data_flg == fig_flg:
            #         # data_flg = False
            #         tmp_arr = []
            #         for v_tmp in v_arr:
            #             if len(v_tmp) > 4:
            #                 v_t = v_tmp.split('-')
            #                 tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
            #             else:
            #                 tmp_arr.append(v_tmp)
            #         data['data'].append(v_arr)
            #         data['test_size'].append(test_size)
            #     v = change_line(v_arr, k=2, value_type=value_type)
            #
            # elif res_type.upper() == 'appd_all_with_header'.upper():
            #     representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
            #     # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
            #     colors = seaborn_palette(res_type, fig_type='raw')
            #     value_type = 'best+min'
            #     v_arr = [process_value(v[4 + num_feats + 1]), process_value(v[7 + num_feats + 1]),
            #              process_value(v[5 + num_feats + 1]),
            #              process_value(v[8 + num_feats + 1]),
            #              process_samp_value(v[11 + num_feats + 1], value_type=value_type),
            #              process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
            #     if data_flg == fig_flg:
            #         # data_flg = False    # value is negative
            #         tmp_arr = []
            #         try:
            #             for v_tmp in v_arr:
            #                 if len(v_tmp) > 4:
            #                     v_t = v_tmp.split('-')
            #                     tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
            #                 else:
            #                     tmp_arr.append(v_tmp)
            #         except Exception as e:
            #             print(f'Error: {e}, {v_arr}')
            #         data['data'].append(v_arr)
            #         data['test_size'].append(test_size)
            #     v = change_line(v_arr, k=2, value_type=value_type)

            # elif res_type.upper() == 'appd_samp'.upper():
            #     representations = ["SAMP-NUM (wo. header)", "SAMP-NUM (w. header)",
            #                        "SAMP-SIZE (wo. header)", "SAMP-SIZE (w. header)"]
            #     # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
            #     # colors = ['red', 'darkred', 'blue', 'darkblue']  # 'magenta', 'darkmagenta'
            #     colors = seaborn_palette(res_type, fig_type='raw')
            #     value_type = 'best+min+avg'
            #
            #     v_arr = [process_samp_value(v[11], value_type=value_type),
            #              process_samp_value(v[11 + num_feats + 1], value_type=value_type),
            #              process_samp_value(v[12], value_type=value_type),
            #              process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
            #     if data_flg == fig_flg:
            #         # data_flg = False
            #         tmp_arr = []
            #         for v_tmp in v_arr:
            #             if len(v_tmp) > 4:
            #                 v_t = v_tmp.split('-')
            #                 tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
            #             else:
            #                 tmp_arr.append(v_tmp)
            #         data['data'].append(v_arr)
            #         data['test_size'].append(test_size)
            #     v = change_line(v_arr, k=2, value_type=value_type)
            # elif res_type.upper() == 'feature_dimensions'.upper():
            #     # # representations = ["IAT/IAT-FFT","SIZE/SIZE-FFT","STATS", "SAMP-NUM/SAMP-NUM-FFT/SAMP-SIZE"]
            #     # representations = ["IAT"]
            #     # colors = ['red', 'darkred', 'blue', 'darkblue']  # 'magenta', 'darkmagenta'
            #     # value_type = 'wo. head'
            #     # if value_type == 'wo. head':
            #     #     v_arr = [get_dimension_from_cell(v[4]), get_dimension_from_cell(v[5]),
            #     #              get_dimension_from_cell(v[10]),
            #     #              get_dimension_from_cell(v[11], value_type=value_type)]
            #     # elif value_type == 'w. head':
            #     #     v_arr = [get_dimension_from_cell(v[4 + num_feats + 1]),
            #     #              get_dimension_from_cell(v[5 + num_feats + 1]),
            #     #              get_dimension_from_cell(v[10 + num_feats + 1]),
            #     #              get_dimension_from_cell(v[11 + num_feats + 1], value_type=value_type)
            #     #              ]
            #     # v = '&'.join(v_arr)
            #     print(f'{res_type} does not need to show figure.')
            #     return -1
            else:
                v = change_line(v_arr, k=2, value_type=value_type)
        else:
            pass

    return data


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


def plot_bar_difference_seaborn(data={}, show_detectors=['OCSVM', 'AE'],
                                show_datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'],
                                output_file="F1_for_all.pdf", xlim=[-0.1, 1], tab_type=''):
    sns.set_style("darkgrid")
    print(show_detectors)
    # create plots
    num_figs = len(show_detectors)
    if len(show_datasets) <= 7:
        c = 2  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c) + 1  # in each row, it show 4 subplot
            fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    else:
        c = 1  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c)  # in each row, it show 4 subplot
            if "GMM" in ",".join(show_detectors):
                fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            else:
                fig, axes = plt.subplots(r, c, figsize=(18, 18))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')

    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

    new_data = OrderedDict()
    for j, (detector_name, vs_dict) in enumerate(data.items()):
        sub_dataset = []
        new_colors = []
        yerrs = []
        for ind, (dataset_name, (test_size, vs)) in enumerate(vs_dict.items()):
            test_size = sum([int(v) for v in test_size])
            n = 100
            yerr = []
            if dataset_name not in new_data.keys():
                new_data[dataset_name] = {detector_name: {}, 'test_size': [], 'yerr': []}

            if tab_type.upper() == "basic_representation".upper():
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
                # new_colors = ['b', 'r']
                # new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                # yerr = [1 / np.sqrt(test_size)] * 2

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
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 3

            elif tab_type.upper() == "effect_size".upper():
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

                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 2
            elif tab_type.upper() == "effect_header".upper():
                features = ['STATS (wo. header)', 'STATS (w. header)', 'IAT+SIZE (wo. header)', 'IAT+SIZE (w. header)',
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
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()

                A = 'IAT+SIZE (wo. header)'
                B = 'IAT+SIZE (w. header)'
                pre_aucs = [float(vs[f_dict[A]]), 0]
                aucs = [float(vs[f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][repres_pair] = diff[0]

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

                yerr = [1 / np.sqrt(test_size)] * 3

            new_data[dataset_name]['test_size'] = test_size
            new_data[dataset_name]['yerr'] = yerr

            yerrs.append(yerr)
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1
        print(f'{detector_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'repres', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('before yerrs:', yerrs)
        new_yerrs = []
        # yerrs=list(chain.from_iterable(yerrs))
        yerrs = np.asarray(yerrs)
        for c_tmp in range(yerrs.shape[1]):  # extend by columns
            new_yerrs.extend(yerrs[:, c_tmp])
        print('new_yerrs:', new_yerrs)
        g = sns.barplot(y="diff", x='dataset', hue='repres', data=df, palette=new_colors, ci=None,
                        capsize=.2, ax=axes[t, j % c])
        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())

            num_bars = df['repres'].nunique()

            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=new_yerrs, fmt='none', c='b', capsize=3)

        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        font_size = 20
        g.set_ylabel('AUC difference', fontsize=font_size + 4)
        if r > 1:
            # if c==1:
            #     if j < len(show_detectors) - 1:
            #         g.set_xticklabels([])
            #     else:
            #         g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")

            if j < len(show_detectors) - c:
                g.set_xticklabels([])
            else:
                g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")

        else:
            g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        g.set_yticks(y_v)  # set value locations in y axis
        g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
                          fontsize=font_size + 6)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        if j % c != 0:
            # g.get_yaxis().set_visible(False)
            g.set_yticklabels(['' for v_tmp in y_v])
            g.set_ylabel('')

        g.set_title(detector_name, fontsize=font_size + 8)
        g.get_legend().set_visible(False)

    # # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    fig.legend(handles, labels, loc='lower center',  # upper left
               ncol=3, prop={'size': font_size - 2})  # l

    plt.tight_layout()

    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.35)
        else:
            if c == 1 and len(show_detectors) == 4:
                plt.subplots_adjust(bottom=0.1)
            else:
                plt.subplots_adjust(bottom=0.18)
    except Warning as e:
        raise ValueError(e)

    plt.savefig(output_file)  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})


def dat2tables(data, out_file='.txt'):
    features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP-SIZE']

    outs = {}
    for j, (detector_name, vs_dict) in enumerate(data.items()):
        sub_dataset = []

        for ind, (dataset_name, (test_size, vs)) in enumerate(vs_dict.items()):
            # test_size = int(test_size[0]) + int(test_size[1])    # test_size=('100', '100')
            f_dict = dict(zip(features, [i for i in range(len(features))]))
            tmp = [dataset_name]
            for feat in features:
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
        line = 'Detector & Dataset & ' + ' & '.join(features) + '\n\\toprule\n'
        f.write(line)
        for detector_name, vs_lst in outs.items():
            line = f'{detector_name}'
            for vs in vs_lst:
                line += ' & ' + ' & '.join(vs) + '\\\\ \n \\cmidrule{2-8}' + '\n'

            line += '\midrule' + '\n'
            f.write(line)
        f.write('\\bottomrule\n')


def xlsx2tables(results, name, gs, datasets, methods, res_type='effect_size', out_file=''):
    with open(out_file, 'w') as out_f:

        needed_results = OrderedDict()
        for detector_name in methods:
            for i, (sheet_name, value_lst) in enumerate(results.items()):
                if detector_name == sheet_name:
                    print(detector_name)
                    data = parse_data(value_lst, gs, datasets, detector_name, res_type)
                    needed_results[detector_name] = data

        ### get tables
        out_file = os.path.dirname(out_file) + f"/{name}.txt"
        print(out_file)
        dat2tables(needed_results, out_file)

    return out_file


def xlsx2figs(results, name, gs, datasets, methods, res_type='basic_representation', out_file=''):
    with open(out_file, 'w') as out_f:

        needed_results = OrderedDict()
        for detector_name in methods:
            for i, (sheet_name, value_lst) in enumerate(results.items()):
                if detector_name == sheet_name:
                    print(detector_name)
                    data = parse_data(value_lst, gs, datasets, detector_name, res_type)
                    needed_results[detector_name] = data

        ### get figure
        out_file = os.path.dirname(out_file) + f"/{name}.pdf"
        print(out_file)
        plot_bar_difference_seaborn(needed_results, show_detectors=methods, show_datasets=datasets,
                                    tab_type=res_type, output_file=out_file)
    return out_file


def _main(in_dir, results, gses=[True, False],
          datasets_lst=[['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']],
          methods_lst=[['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]):
    for gs in gses:
        for datasets in datasets_lst:
            for methods in methods_lst:
                for res_type in ['basic_representation', 'effect_size', 'effect_header']:
                    tmp = 'best' if gs else 'default'
                    name = '_'.join(methods) + f'-{len(datasets)}-' + tmp + '-' + res_type
                    out_file = f'{in_dir}/res/{name}-latex_tables_figs.txt'  # for main paper results
                    check_n_generate_path(out_file)
                    print(f'\n\n******************\n{out_file}')
                    try:
                        xlsx2tables(results, name, gs, datasets, methods, 'effect_size', out_file)
                        xlsx2figs(results, name, gs, datasets, methods, res_type, out_file)
                    except Exception as e:
                        print('Error: ', e)
                        traceback.print_exc()
                    # break


def main(root_dir='examples/reprst'):
    in_dir = f'{root_dir}/out/report/reprst_srcip/20201227'
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
    ### Get the results on part of datasets and algorithms
    # 1. datasets: [UNB(PC1), UNB(PC4), CTU, MAWI, TV&RT, SFrig, and BSTch]
    # algorithms: [OCSVM, IF, AE, KDE] and [GMM, PCA]
    ### parse the results from the xlsx
    print('parse xlsx.')
    results = parse_xlsx(xlsx_file)
    gses = [True, False]
    datasets_lst = [['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']]
    methods_lst = [['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)

    # 2. datasets: ['UNB(PC2)', 'UNB(PC3)',  'UNB(PC5)', 'GHom', 'SCam']
    print('parse xlsx.')  # reparse xlsx to avoid any changes that may happen in previous operations.
    results = parse_xlsx(xlsx_file)
    datasets_lst = [['UNB(PC2)', 'UNB(PC3)', 'UNB(PC5)', 'GHom', 'SCam']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)

    ########################################################################################################
    ### 3. Get the results on all datasets and part of algorithms ([OCSVM, IF, AE, KDE] and [GMM, PCA]).
    print('parse xlsx.')
    results = parse_xlsx(xlsx_file)  # reparse xlsx to avoid any changes that may happen in previous operations.
    gses = [True, False]
    datasets_lst = [['UNB(PC1)', 'UNB(PC2)', 'UNB(PC3)', 'UNB(PC4)', 'UNB(PC5)', 'CTU', 'MAWI', 'TV&RT', 'GHom',
                     'SCam', 'SFrig', 'BSTch']]
    methods_lst = [['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]
    _main(in_dir, results, gses, datasets_lst, methods_lst)

    # ########################################################################################################
    # ### 4. Get the results on all datasets and algorithms ([OCSVM, IF, AE, KDE, GMM, PCA]).
    ### parse the results from the xlsx
    # print('parse xlsx.')
    # results = parse_xlsx(xlsx_file) # reparse xlsx to avoid any changes that may happen in previous operations.
    # gses = [True, False]
    # datasets_lst = [['UNB(PC1)', 'UNB(PC2)',  'UNB(PC3)', 'UNB(PC4)',  'UNB(PC5)', 'CTU', 'MAWI', 'TV&RT', 'GHom',
    # 'SCam', 'SFrig', 'BSTch']]
    # methods_lst = [['OCSVM', 'IF', 'AE', 'KDE', 'GMM', 'PCA']]
    # _main(in_dir, results, gses, datasets_lst, methods_lst)


if __name__ == '__main__':
    main()
