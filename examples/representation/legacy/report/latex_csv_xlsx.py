"""Data postprocess including data format transformation, highlight data

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import os, sys
import traceback

lib_path = os.path.abspath('.')
sys.path.append(lib_path)
print(f"add \'{lib_path}\' into sys.path: {sys.path}")
#
# import matplotlib
# matplotlib.use('TkAgg')     # pycharm can't use subplots_adjust() to show the legend and modified subplots correctly.

import copy
from operator import itemgetter

import xlsxwriter
from matplotlib import rcParams
from pandas import ExcelWriter

from itod.utils.tool import *
from itod.visual.visualization import plot_bar
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import ListedColormap
from itertools import chain
from fractions import Fraction

import seaborn as sns

# sns.set_style("darkgrid")

# colorblind for diff
sns.set_palette("bright")  # for feature+header
sns.palplot(sns.color_palette())


# plt.show()
# plt.close()


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


def txt2csv(input_file='xxx.txt', output_file='xxx.csv'):
    with open(input_file, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(output_file, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('title', 'intro'))
            writer.writerows(lines)

    return output_file


def csv2latex_previous(input_file='', tab_latex='', caption='', do_header=False, previous_result=True,
                       num_feat=4, verbose=1):
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
                smart_tv_line_flg = False
                continue
            if gs_false:
                if do_header:
                    if num_feat == 3:
                        arr_tmp = [arr[11], arr[14], arr[15]]
                    else:  # num_feat ==4
                        arr_tmp = [arr[12], arr[15], arr[16], arr[17]]
                else:
                    if num_feat == 3:
                        arr_tmp = [arr[4], arr[7], arr[8]]
                    else:
                        arr_tmp = [arr[4], arr[7], arr[8], arr[9]]

                arr_tmp = change_line(arr_tmp, k=k)
                # tab_latex_true[8 + t] = tab_latex_true[8+t].format(arr_tmp)
                for _ in range(len(tab_latex_false[(8 + t):-1])):
                    if '\midrule' in tab_latex_false[8 + t]:
                        t += 1
                    else:
                        break
                if previous_result:
                    if 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1' in line:
                        smart_tv_line = arr_tmp
                        smart_tv_line_flg = True
                        continue
                    if 'Smart TV' in tab_latex_false[8 + t] and smart_tv_line_flg:
                        tab_latex_false[8 + t] = tab_latex_false[8 + t].replace('{v}', smart_tv_line)
                        t += 1
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
                smart_tv_line_flg = False
                t = 0
                continue

            if gs_ture:
                if do_header:
                    if num_feat == 3:
                        arr_tmp = [arr[11], arr[14], arr[15]]
                    else:  # num_feat ==4
                        arr_tmp = [arr[12], arr[15], arr[16], arr[17]]
                else:
                    if num_feat == 3:
                        arr_tmp = [arr[4], arr[7], arr[8]]
                    else:
                        arr_tmp = [arr[4], arr[7], arr[8], arr[9]]

                arr_tmp = change_line(arr_tmp, k=k)

                # tab_latex_true[8 + t] = tab_latex_true[8+t].format(arr_tmp)
                for _ in range(len(tab_latex_true[(8 + t):-1])):
                    if '\midrule' in tab_latex_true[8 + t]:
                        t += 1
                    else:
                        break
                if previous_result:
                    if 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1' in line:
                        smart_tv_line = arr_tmp
                        smart_tv_line_flg = True
                        continue
                    if 'Smart TV' in tab_latex_false[8 + t] and smart_tv_line_flg:
                        tab_latex_true[8 + t] = tab_latex_true[8 + t].replace('{v}', smart_tv_line)
                        t += 1
                tab_latex_true[8 + t] = tab_latex_true[8 + t].replace('{v}', arr_tmp)
                t += 1

            if verbose:
                print(f'{i}, {line.strip()}')
        if verbose:
            print('+++ gs:True\n')
        print('\n'.join(tab_latex_true))

        return tab_latex_false, tab_latex_true


@func_notation
def write2csv(result_dict, detector_name='GMM', ws_name='subflows', output_file='example.csv'):
    # value_lst = ['DS10_UNB_IDS/DS11-srcIP_192.168.10.5', 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
    #              'DS10_UNB_IDS/DS13-srcIP_192.168.10.9', 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
    #              'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
    #
    #              'DS30_OCS_IoT/DS31-srcIP_192.168.0.13',
    #
    #              'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
    #
    #              'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
    #
    #              'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
    #
    #              'DS60_UChi_IoT/DS61-srcIP_192.168.143.20', 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
    #              'DS60_UChi_IoT/DS63-srcIP_192.168.143.43', 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
    #
    #              ]  # 'DEMO_IDS/DS-srcIP_192.168.10.5'
    value_lst = [
        'UNB/CICIDS_2017/pc_192.168.10.5',
        'UNB/CICIDS_2017/pc_192.168.10.8',
        'UNB/CICIDS_2017/pc_192.168.10.9',
        'UNB/CICIDS_2017/pc_192.168.10.14',
        'UNB/CICIDS_2017/pc_192.168.10.15',
        #
        'CTU/IOT_2017/pc_10.0.2.15',
        #
        'MAWI/WIDE_2019/pc_202.171.168.50',
        # 'MAWI/WIDE_2020/pc_203.78.7.165',
        # 'MAWI/WIDE_2020/pc_202.119.210.242',
        #
        'UCHI/IOT_2019/smtv_10.42.0.1',
        #
        'UCHI/IOT_2019/ghome_192.168.143.20',
        'UCHI/IOT_2019/scam_192.168.143.42',
        'UCHI/IOT_2019/sfrig_192.168.143.43',
        'UCHI/IOT_2019/bstch_192.168.143.48'

    ]  # 'DEMO_IDS/DS-srcIP_192.168.10.5'

    with open(output_file, 'w') as out_hdl:

        out_hdl.write('\n')

        gs = 'gs:False'
        line = f'{detector_name}, Dataset, Train set, Test set, IAT, SIZE, IAT+SIZE, FFT_IAT, FFT_SIZE, FFT_IAT+SIZE, ' \
               f'STAT, SAMP_NUM, SAMP_SIZE, SAMP_NUM+SIZE, FFT_SAMP_NUM, FFT_SAMP_SIZE, FFT_SAMP_NUM+SIZE, ,' \
               'IAT, SIZE, IAT+SIZE, FFT_IAT, FFT_SIZE, FFT_IAT+SIZE, STAT, SAMP_NUM, SAMP_SIZE, SAMP_NUM+SIZE, ' \
               'FFT_SAMP_NUM, FFT_SAMP_SIZE, FFT_SAMP_NUM+SIZE'
        out_hdl.write(line + '\n')
        line = f'{gs}, subflow_interval = np.quantile(flow_durations . q=-),  , , without header features, , , , , , , , , , , , , , ' \
               'with header features, , , , , ,, ,,,,,,'
        out_hdl.write(line + '\n')

        for i, dataset_name in enumerate(value_lst):
            line = ''
            if dataset_name.startswith('UNB/CICIDS_2017'):
                data_cat = 'AGMT'
            else:
                data_cat = 'INDV'
            try:
                sf_key = get_sf_key(result_dict[detector_name][gs], v='sf:True', header='hdr:False',
                                    data_cat=data_cat, dataset_name=dataset_name)
                # print('sf_key', sf_key, dataset_name)
                value = result_dict[detector_name][gs][sf_key]['hdr:False'][data_cat][dataset_name]
                tmp_v = ','.join(value[0].split(',')[1:])
                line = f'{i}-{sf_key}, {dataset_name},{tmp_v}, ,'
            except KeyError as e:
                print(f'i:{i}, {gs}, KeyError:{e} ')
                tmp_v = '-,' * (2 + 13)  # train, test, iat, size, iat_size, fft, stats, baseline1, baseline2, feat_dim,
                line = f'{i}, {dataset_name},{tmp_v} ,'
            try:
                # sf_key = get_sf_key(list(result_dict[detector_name][gs].keys()), v='sf:True')
                sf_key = get_sf_key(result_dict[detector_name][gs], v='sf:True', header='hdr:True',
                                    data_cat=data_cat, dataset_name=dataset_name)
                value = result_dict[detector_name][gs][sf_key]['hdr:True'][data_cat][dataset_name]
                tmp_v = ','.join(value[0].split(',')[3:])  # key, 0:20000,0:14 1:14, iat, iat_size, ...
                line += tmp_v
            except KeyError as e:
                print(f'i:{i}, {gs}, KeyError:{e} ')
                tmp_v = '-,' * 13
                line += tmp_v
            if line != '':
                out_hdl.write(line + '\n')

        out_hdl.write('\n')

        out_hdl.write(','.join([''] * len(line.split(','))) + '\n')

        gs = 'gs:True'
        line = f'{detector_name}, Dataset, Train set, Test set, IAT, SIZE, IAT+SIZE, FFT_IAT, FFT_SIZE, FFT_IAT+SIZE, ' \
               f'STAT, SAMP_NUM, SAMP_SIZE, SAMP_NUM+SIZE, FFT_SAMP_NUM, FFT_SAMP_SIZE, FFT_SAMP_NUM+SIZE, , ' \
               'IAT, SIZE, IAT+SIZE, FFT_IAT, FFT_SIZE, FFT_IAT+SIZE, STAT, SAMP_NUM, SAMP_SIZE, SAMP_NUM+SIZE, ' \
               'FFT_SAMP_NUM, FFT_SAMP_SIZE, FFT_SAMP_NUM+SIZE'
        out_hdl.write(line + '\n')
        line = f'{gs}, subflow_interval = np.quantile(flow_durations . q=-),  , , without header features, , , , , , , , , , ,,,,' \
               'with header features, , , , , ,, ,,,,,,'
        out_hdl.write(line + '\n')

        for i, dataset_name in enumerate(value_lst):
            line = ''
            if dataset_name.startswith('UNB/CICIDS_2017'):
                data_cat = 'AGMT'
            else:
                data_cat = 'INDV'
            try:
                # sf_key = get_sf_key(list(result_dict[detector_name][gs].keys()), v='sf:True')
                sf_key = get_sf_key(result_dict[detector_name][gs], v='sf:True', header='hdr:False',
                                    data_cat=data_cat, dataset_name=dataset_name)
                # print('sf_key', sf_key, dataset_name, result_dict[detector_name][gs][sf_key]['header:False'][data_cat])
                value = result_dict[detector_name][gs][sf_key]['hdr:False'][data_cat][dataset_name]
                tmp_v = ','.join(value[0].split(',')[1:])
                line = f'{i}-{sf_key}, {dataset_name},{tmp_v}, ,'
            except KeyError as e:
                print(f'i:{i}, {gs}, KeyError:{e} ')
                tmp_v = '-,' * (2 + 13)
                line = f'{i}, {dataset_name},{tmp_v} ,'
            try:
                # sf_key = get_sf_key(list(result_dict[detector_name][gs].keys()), v='sf:True')
                sf_key = get_sf_key(result_dict[detector_name][gs], v='sf:True', header='hdr:True',
                                    data_cat=data_cat, dataset_name=dataset_name)
                value = result_dict[detector_name][gs][sf_key]['hdr:True'][data_cat][dataset_name]
                tmp_v = ','.join(value[0].split(',')[3:])
                line += tmp_v
            except KeyError as e:
                print(f'i:{i}, {gs}, KeyError:{e} ')
                tmp_v = '-,' * 13
                line += tmp_v
            if line != '':
                out_hdl.write(line + '\n')

    return output_file


@func_notation
def csv2xlsx(filename, detector_name='OCSVM', output_file='example.xlsx'):
    read_file = pd.read_csv(filename, header=0, index_col=False)  # index_col=False: not use the first columns as index
    read_file.to_excel(output_file, sheet_name=detector_name, index=0, header=True)

    return output_file


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


tab_latex_3 = [
    "\\begin{table}[hbt!]",
    "\\centering",
    "\\label{{v}}",
    "\\caption{{v}}",
    "\\begin{tabular}{lllll}",
    "\\toprule",
    "Source & Device & IAT & STATS & $SAMP_{num_{pkts}}$ \\\ ",
    "% \cmidrule{1-4}",
    "\midrule",
    "\\multirow{5}{*}{UNB IDS} & PC &  {v} \\\ ",
    " & PC &  {v} \\\ ",
    " & PC &  {v} \\\ ",
    " & PC &  {v} \\\ ",
    " & PC & {v} \\\ ",
    "\midrule",
    "HCR IoT & SKT camera & {v} \\\ ",
    "\midrule",
    "CTU IoT & PC & {v}\\\ ",
    "\midrule",
    "MAWI WIDE & PC &  {v}\\\ ",
    "\midrule",
    "\multirow{5}{*}{Private IoT} & Smart TV & {v} \\\ ",
    " & Google home & {v} \\\ ",
    " & Samsung fridge & {v} \\\ ",
    " & Samsung camera &  {v} \\\ ",
    " & Bose soundtouch & {v}  \\\ ",
    "\\bottomrule",
    "\end{tabular}",
    "\end{table}",
]

tab_latex_7 = [
    "\\begin{table}[hbt!]",
    "\\centering",
    "\\label{{v}}",
    "\\caption{{v}}",
    "\\begin{tabular}{lllllllll}",
    "\\toprule",
    "Source & Device & IAT &SIZES & IAT+SIZES & FFT & STATS & $SAMP_{num_{pkts}}$ & $SAMP_{size}$ \\\ ",
    "% \cmidrule{1-4}",
    "\midrule",
    "\\multirow{5}{*}{UNB IDS} & PC1 &  {v} \\\ ",
    " & PC2 &  {v} \\\ ",
    " & PC3 &  {v} \\\ ",
    " & PC4 &  {v} \\\ ",
    " & PC5 & {v} \\\ ",
    "\midrule",
    "HCR IoT & SKT camera & {v} \\\ ",
    "\midrule",
    "CTU IoT & PC & {v}\\\ ",
    "\midrule",
    "MAWI WIDE & PC &  {v}\\\ ",
    "\midrule",
    "\multirow{5}{*}{Private IoT} & Smart TV & {v} \\\ ",
    " & Google home & {v} \\\ ",
    " & Samsung fridge & {v} \\\ ",
    " & Samsung camera &  {v} \\\ ",
    " & Bose soundtouch & {v}  \\\ ",
    "\\bottomrule",
    "\end{tabular}",
    "\end{table}",
]


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


def tab_temple(name='', caption='', cols_format='', label='', cols=[]):
    """
    ref: https://tex.stackexchange.com/questions/23385/table-numbering-mismatch-in-caption-and-in-text
    Always put \label after \caption to get the consistent table number format

    Parameters
    ----------
    name
    caption
    cols_format
    label
    cols

    Returns
    -------

    """
    # num_cols = 'l'+'l' * (len(cols))
    cols_str = " & ".join(cols)

    tab_latex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{" + f"{caption}" + "}",
        "\\label{tab:" + f"{label}" + "}",
        "\\begin{tabular}{" + f"{cols_format}" + "}",
        "\\toprule",
        f"Device & {cols_str} \\\ ",
        "% \cmidrule{1-4}",
        "\midrule",
        "\\multirow{5}{*}\ PC1 & {v} \\\ ",
        "\ PC2 & {v} \\\ ",
        "\ PC3 & {v} \\\ ",
        "\ PC4 & {v} \\\ ",
        "\ PC5 & {v} \\\ ",
        "\midrule",
        # "HCR IoT & SKT camera & {v} \\\ ",
        # "\midrule",
        "\ 2Rsps & {v}\\\ ",
        "\midrule",
        "\ 2PCs & {v}\\\ ",
        "\midrule",
        "\multirow{5}{*}\ TV\&RT  & {v} \\\ ",
        "\ GHom & {v} \\\ ",
        "\ SCam & {v} \\\ ",
        "\ SFrig & {v} \\\ ",
        "\ BSTch & {v}  \\\ ",
        "\\bottomrule",
        "\end{tabular}",
        "\end{table}",
    ]
    return tab_latex


def comb_tab_temple(name='', caption='', cols_format='', label='', detector_cols=[], feature_cols=[],
                    fig_flg='default'):
    """
    ref: https://tex.stackexchange.com/questions/23385/table-numbering-mismatch-in-caption-and-in-text
    Always put \label after \caption to get the consistent table number format

    # add the following command at the begining of the latex
    # latex new command: \newcommand{\Cell}[1]{\begin{tabular}{@{}c}{\Centerstack[c]{#1}}\end{tabular}}
    # https://tex.stackexchange.com/questions/273439/i-want-to-create-a-new-command-for-making-a-table-parameters-in-the-command-wou

    Parameters
    ----------
    name
    caption
    cols_format
    label
    cols

    Returns
    -------

    """
    # num_cols = 'l'+'l' * (len(cols))
    v_str = "\multicolumn{" + f"{len(feature_cols)}" + "}{c|} "
    detector_cols_str = " & ".join([v_str + '{' + v + '}' for v in detector_cols])
    new_feat_colos = []
    for feat in feature_cols:
        if 'SAMP-' in feat:
            if len(feat) > 5:
                feat = "\\\\".join(textwrap.wrap(feat, width=5))
            # feat = '\\begin{tabular}{@{}c}{\shortstack[c]{' + feat + '}} \end{tabular}'  # @{}: with indent in the cell
            feat = '\\Cell{' + feat + '}'  # @{}: with indent in the cell
        else:
            if len(feat) > 5:
                feat = "\\\\".join(textwrap.wrap(feat, width=5))
            # feat = '\\begin{tabular}{@{}c}{\shortstack[c]{' + feat + '}} \end{tabular}'
            feat = '\\Cell{' + feat + '}'  # @{}: with indent in the cell
        new_feat_colos.append(feat)
    feature_cols_str = (' & '.join(new_feat_colos) + ' &') * len(detector_cols)
    feature_cols_str = feature_cols_str[:-1]
    num_f = len(feature_cols) * len(detector_cols) + 1
    if fig_flg == 'default':
        tab_latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{" + f"{caption}" + "}",
            "\\label{tab:" + f"{label}" + "}",
            # "\\begin{widepage}",
            # "\\begin{tabular}{" + f"{cols_format}" + "}",
            "\\begin{tabulary}{0.99\\textwidth}{" + f"{cols_format}" + "}",  # auto indenet the cell with tabulary
            "\\toprule",
            "\\multirow{2}{*}{~\\rule{0pt}{2.7ex} Dataset} & " + f" {detector_cols_str} \\\ ",
            # Inserting a small vertical space in a table
            # & {OCSVM} & \multicolumn{3}{c|}{IF} & \multicolumn{3}{c|}{AE}
            "\cmidrule{2-" + f"{num_f}" + "}",
            f" & {feature_cols_str}   \\\ ",
            "\midrule",
            "\ UNB(PC1) & {v}  \\\ ",
            "\ UNB(PC4) & {v} \\\ ",
            "\midrule",
            "\ CTU & {v}\\\ ",
            "\midrule",
            "\ MAWI & {v}\\\ ",
            "\midrule",
            "\ TV\&RT  & {v}\\\ ",
            "\midrule",
            "\ SFrig & {v} \\\ ",
            "\ BSTch & {v}  \\\ ",
            "\\bottomrule",
            "\end{tabulary}",
            # "\end{widepage}",
            "\end{table}",
        ]
    else:
        tab_latex = [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{" + f"{caption}" + "}",
            "\\label{tab:" + f"{label}" + "}",
            # "\\begin{widepage}",
            # "\\begin{tabular}{" + f"{cols_format}" + "}",
            "\\begin{tabulary}{0.99\\textwidth}{" + f"{cols_format}" + "}",  # auto indenet the cell with tabulary
            "\\toprule",
            "\\multirow{2}{*}{~\\rule{0pt}{2.7ex} Dataset} & " + f" {detector_cols_str} \\\ ",
            # & {OCSVM} & \multicolumn{3}{c|}{IF} & \multicolumn{3}{c|}{AE}
            "\cmidrule{2-" + f"{num_f}" + "}",
            f" & {feature_cols_str}   \\\ ",
            "\midrule",
            "\ UNB(PC2) & {v}  \\\ ",
            "\ UNB(PC3) & {v} \\\ ",
            "\ UNB(PC5) & {v} \\\ ",
            "\midrule",
            "\ GHom & {v}\\\ ",
            "\midrule",
            "\ SCam & {v}\\\ ",
            "\\bottomrule",
            "\end{tabulary}",
            # "\end{widepage}",
            "\end{table}",
        ]
    return tab_latex


def comb_all_needed_data_tab_temple(name='', caption='', cols_format='', label='', detector_cols=[], feature_cols=[],
                                    fig_flg='default', appendix_flg=False):
    """
    ref: https://tex.stackexchange.com/questions/23385/table-numbering-mismatch-in-caption-and-in-text
    Always put \label after \caption to get the consistent table number format

    # add the following command at the begining of the latex
    # latex new command: \newcommand{\Cell}[1]{\begin{tabular}{@{}c}{\Centerstack[c]{#1}}\end{tabular}}
    # https://tex.stackexchange.com/questions/273439/i-want-to-create-a-new-command-for-making-a-table-parameters-in-the-command-wou

    Parameters
    ----------
    name
    caption
    cols_format
    label
    cols

    Returns
    -------

    """
    # num_cols = 'l'+'l' * (len(cols))
    v_str = "\multicolumn{" + f"{len(feature_cols)}" + "}{c|} "
    detector_cols_str = " & ".join([v_str + '{' + v + '}' for v in detector_cols])
    new_feat_colos = []
    for feat in feature_cols:
        if 'SAMP-' in feat:
            if len(feat) > 5:
                feat = "\\\\".join(textwrap.wrap(feat, width=5))
            # feat = '\\begin{tabular}{@{}c}{\shortstack[c]{' + feat + '}} \end{tabular}'  # @{}: with indent in the cell
            feat = '\\Cell{' + feat + '}'  # @{}: with indent in the cell
        else:
            if len(feat) > 5:
                feat = "\\\\".join(textwrap.wrap(feat, width=5))
            # feat = '\\begin{tabular}{@{}c}{\shortstack[c]{' + feat + '}} \end{tabular}'
            feat = '\\Cell{' + feat + '}'  # @{}: with indent in the cell
        new_feat_colos.append(feat)
    feature_cols_str = (' & '.join(new_feat_colos) + ' &')
    feature_cols_str = feature_cols_str[:-1]
    num_f = len(feature_cols) * len(detector_cols) + 1

    tab_latex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{" + f"{caption}" + "}",
        "\\label{tab:" + f"{label}" + "}",
        # "\\begin{widepage}",
        # "\\begin{tabular}{" + f"{cols_format}" + "}",
        "\\begin{tabulary}{0.99\\textwidth}{" + f"{cols_format}" + "}",  # auto indent the cell with tabulary
        "\\toprule",
        "~\\rule{0pt}{2.7ex} Detector & Dataset &" + f" {feature_cols_str} \\\ ",
        # # Inserting a small vertical space in a table
        # # & {OCSVM} & \multicolumn{3}{c|}{IF} & \multicolumn{3}{c|}{AE}
        # "\cmidrule{2-" + f"{num_f}" + "}",
        # f" & {feature_cols_str}   \\\ ",
    ]
    if not appendix_flg:
        if fig_flg == 'default':
            num_data = 7
            for i, detector, in enumerate(detector_cols):
                tab_latex += ["\midrule",
                              "\\multirow{" + str(num_data) + "}{*}{~\\rule{0pt}{2.7ex}" + str(detector) + "} " +
                              "&UNB(PC1) & {v}  \\\ ",
                              "&UNB(PC4) & {v} \\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&CTU & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&MAWI & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&TV\&RT & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&SFrig & {v} \\\ ",
                              "&BSTch & {v}  \\\ ", ]
        else:
            num_data = 5
            for i, detector, in enumerate(detector_cols):
                tab_latex += ["\midrule",
                              "\\multirow{" + str(num_data) + "}{*}{~\\rule{0pt}{2.7ex}" + str(detector) + "} " +
                              "&UNB(PC2) & {v}  \\\ ",
                              "&UNB(PC3) & {v} \\\ ",
                              "&UNB(PC5) & {v} \\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&GHom & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&SCam & {v}\\\ ",
                              ]

    else:
        if fig_flg == 'default':
            num_data = 7
            for i, detector, in enumerate(detector_cols):
                tab_latex += ["\midrule",
                              "\\multirow{" + str(num_data) + "}{*}{" + str(detector) + "} " +  # ~\\rule{0pt}{2.7ex}
                              "&UNB(PC1) & {v}  \\\ ",
                              "&UNB(PC2) & {v}  \\\ ",
                              "&UNB(PC3) & {v} \\\ ",
                              "&UNB(PC4) & {v} \\\ ",
                              "&UNB(PC5) & {v} \\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&CTU & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&MAWI & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&TV\&RT & {v}\\\ ",
                              "\cmidrule{2-" + str(2 + len(feature_cols)) + "}",
                              "&GHom & {v}\\\ ",
                              "&SCam & {v}\\\ ",
                              "&SFrig & {v} \\\ ",
                              "&BSTch & {v}  \\\ ", ]

    tab_latex += ["\\bottomrule",
                  "\end{tabulary}",
                  # "\end{widepage}",
                  "\end{table}", ]

    return tab_latex


def seaborn_palette(feat_type='', fig_type='raw'):
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
    unique_flg = False
    if unique_flg:  # each feature has one color
        if feat_type == "basic_representation".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'IAT': raw_feat['IAT'], 'IAT-FFT': colors_dark[C_IAT],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-NUM-FFT': colors_deep[C_SAMP_NUM]
                          }  # red
            elif fig_type == 'diff'.upper():
                # 'IAT' vs. IAT-FFT
                colors = {'IAT vs. IAT-FFT': colors_colorblind[C_IAT],  # green
                          'SAMP-NUM vs. SAMP-NUM-FFT': colors_colorblind[C_SAMP_NUM]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

        elif feat_type == "effect_size".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'SIZE': raw_feat['SIZE'], 'IAT': raw_feat['IAT'],
                          'IAT+SIZE': colors_dark[C_IAT],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-SIZE': raw_feat['SAMP-SIZE']
                          }  # red
            elif fig_type == 'diff'.upper():
                colors = {'IAT vs. IAT+SIZE': colors_colorblind[C_IAT],  # green
                          'SAMP-NUM vs. SAMP-SIZE': colors_colorblind[C_SAMP_SIZE]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)
        elif feat_type == "effect_header".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS (wo. header)': raw_feat['STATS'], 'STATS (w. header)': colors_deep[C_STATS],
                          'IAT+SIZE (wo. header)': colors_dark[C_IAT], 'IAT+SIZE (w. header)': colors_deep[C_IAT],
                          # green
                          'SAMP-SIZE (wo. header)': raw_feat['SAMP-SIZE'],
                          'SAMP-SIZE (w. header)': colors_deep[C_SAMP_SIZE]}  # red
            elif fig_type == 'diff'.upper():
                colors = {'STATS (wo. header) vs. STATS (w. header)': colors_colorblind[C_STATS],
                          'IAT+SIZE (wo. header) vs. IAT+SIZE (w. header)': colors_colorblind[C_SIZE],  # green
                          'SAMP-SIZE (wo. header) vs. SAMP-SIZE (w. header)': colors_bright[6]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

    else:  # for the paper
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


def csv2latex_tab(detector, tab_latex=tab_latex_7, input_file='', output_file='', verbose=0):
    with open(output_file, 'w') as f:
        tab_false, tab_true = csv2latex(input_file=input_file,
                                        tab_latex=tab_latex, caption=detector, do_header=False,
                                        verbose=verbose)

        f.write(detector + ':' + input_file + '\n')
        for line in tab_false:
            f.write(line + '\n')

        for line in tab_true:
            f.write(line + '\n')

        tab_false, tab_true = csv2latex(input_file=input_file,
                                        tab_latex=tab_latex, caption=detector + ' header:True', do_header=True,
                                        verbose=verbose)

        f.write(detector + ':' + input_file + '\n')
        for line in tab_false:
            f.write(line + '\n')

        for line in tab_true:
            f.write(line + '\n')

        f.write('\n')

    return output_file


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


def append2file(f, tab_type='', detector_name='', gs=False, value_lst=[], latex='table', fig_flg='default',
                output_file=''):
    if tab_type.upper() == 'basic_representation'.upper():
        # cols = ["IAT", "IAT-FFT", "STATS",  "SAMP-NUM", "SAMP-NUM-FFT"]  # all without header
        cols = ["\\shortstack[l]{IAT}", "\\shortstack[l]{IAT-FFT}", "\\shortstack[l]{STATS}",
                "\\begin{tabular}{l}{\\shortstack[l]{SAMP-NUM \\\\\# (best - min) $q_s$}} \\end{tabular}",
                "\\begin{tabular}{l}{\\shortstack[l]{SAMP-NUM-FFT \\\\\# (best - min) $q_s$}} \\end{tabular}"]  # all without header
        cols_format = 'l' + 'c' * (len(cols))
        caption = f"{detector_name} with xxx parameters on basic representations"
    elif tab_type.upper() == 'effect_size'.upper():
        # cols = ["IAT", "SIZE", "IAT+SIZE", "STATS",  "SAMP-NUM", "SAMP-SIZE"]  # all without header
        cols = ["\\shortstack[l]{IAT}", "\\shortstack[l]{SIZE}", "\\shortstack[l]{IAT+SIZE}", "\\shortstack[l]{STATS}",
                "\\begin{tabular}{l}{\\shortstack[l]{SAMP-NUM \\\\\# (best - min) $q_s$}} \\end{tabular}",
                "\\begin{tabular}{l}{\\shortstack[l]{SAMP-SIZE \\\\\# (best - min) $q_s$}} \\end{tabular}"]  # all without header
        # caption = ' on SIZE representations'
        cols_format = 'l' + 'c' * (len(cols))
        caption = f"Effect of Size: {detector_name} with xxx parameters"
    elif tab_type.upper() == 'effect_header'.upper():
        # cols = ["IAT+SIZE (without header)", "IAT+SIZE\\(with header)",
        #         "SAMP-SIZE\\( without header)",
        #         "SAMP-SIZE\\(with header)", "STATS\\(without header)", "STATS\\(with header)"]
        cols = ["\\begin{tabular}{@{}l}{\\shortstack[c]{IAT+SIZE \\\(wo. header)}} \\end{tabular}",
                "\\begin{tabular}{@{}l}{\\shortstack[c]{IAT+SIZE \\\(w. header)}} \\end{tabular}",
                "\\begin{tabular}{@{}l}{\\shortstack[c]{STATS \\\(wo. header)}} \\end{tabular}",
                "\\begin{tabular}{@{}l}{\\shortstack[c]{STATS \\\(w. header)}} \\end{tabular}",
                # "\\begin{tabular}{@{}l}}{\\shortstack[c]{SAMP-SIZE \\\(wo. header)}} \\end{tabular}",
                # "\\begin{tabular}{@{}l}}{\\shortstack[c]{SAMP-SIZE \\\(w. header)}} \\end{tabular}"
                "\\begin{tabular}{@{}l}{\\shortstack[c]{SAMP-SIZE \\\(wo. header) \\\\\# (best - min) $q_s$}} \\end{tabular}",
                "\\begin{tabular}{@{}l}{\\shortstack[c]{SAMP-SIZE \\\(w. header) \\\\\# (best - min) $q_s$}} \\end{tabular}"
                ]
        cols_format = "p{1.3cm}>{\centering\\arraybackslash}p{1.4cm}>{\centering\\arraybackslash}p{1.4cm}" \
                      ">{\centering\\arraybackslash}p{1.4cm}>{\centering\\arraybackslash}p{1.4cm}" \
                      ">{\centering\\arraybackslash}p{3cm}>{\centering\\arraybackslash}p{3cm}"
        # cols_format = "llllllll"
        caption = f"Effect of Packet Header: {detector_name} with xxx parameters"
    elif tab_type.upper() == 'appd_all_without_header'.upper():  # all without header
        # cols = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
        cols = ["\\shortstack[c]{IAT}", "\\shortstack[c]{IAT-FFT}",
                "\\shortstack[c]{SIZE}", "\\shortstack[c]{SIZE-FFT}",
                "\\begin{tabular}{c}{\\shortstack[c]{SAMP-NUM \\\\\# (best - min) $q_s$}} \\end{tabular}",
                "\\begin{tabular}{c}{\\shortstack[c]{SAMP-SIZE \\\\\# (best - min) $q_s$}} \\end{tabular}"]  # all without header
        cols_format = 'l' + 'c' * (len(cols))
        caption = f"{detector_name} with xxx parameters on representations without header"
    elif tab_type.upper() == 'appd_all_with_header'.upper():
        # cols = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
        cols = ["\\shortstack[c]{IAT}", "\\shortstack[c]{IAT-FFT}",
                "\\shortstack[c]{SIZE}", "\\shortstack[c]{SIZE-FFT}",
                "\\begin{tabular}{c}{\\shortstack[c]{SAMP-NUM \\\\\# (best - min) $q_s$}} \\end{tabular}",
                "\\begin{tabular}{c}{\\shortstack[c]{SAMP-SIZE \\\\\# (best - min) $q_s$}} \\end{tabular}"]  # all with header
        cols_format = 'l' + 'c' * (len(cols))
        caption = f"{detector_name} with xxx parameters on representations with header"
    elif tab_type.upper() == 'appd_samp'.upper():
        # cols = ["SAMP-NUM\\(without header)", "SAMP-NUM\\(with header)", "SAMP-SIZE\\(without header)",
        #         "SAMP-SIZE\\(with header)"]
        cols = [
            "\\begin{tabular}{p{2.5cm}}{\shortstack[c]{SAMP-NUM \\\(wo. header) \\\\\# (best - min - avg)}} \\end{tabular}",
            "\\begin{tabular}{p{2.5cm}}{\shortstack[c]{SAMP-NUM \\\(w. header) \\\\\# (best - min - avg)}} \\end{tabular}",
            "\\begin{tabular}{p{2.5cm}}{\shortstack[c]{SAMP-SIZE \\\(wo. header) \\\\\# (best - min - avg)}} \\end{tabular}",
            "\\begin{tabular}{p{2.5cm}}{\shortstack[c]{SAMP-SIZE \\\(w. header) \\\\\# (best - min - avg)}} \\end{tabular}"]
        cols_format = 'l' + 'c' * (len(cols))
        caption = f"{detector_name} with xxx parameters on SAMP-based representations with/without header"

    elif tab_type.upper() == 'feature_dimensions'.upper():
        cols = [
            "\\begin{tabular}{@{}l}{\shortstack[c]{IAT\\IAT-FFT}} \\end{tabular}",
            "\\begin{tabular}{@{}l}{\shortstack[c]{SIZE\\SIZE-FFT}} \\end{tabular}",
            "\\begin{tabular}{@{}l}{\shortstack[c]{STATS}} \\end{tabular}",
            "\\begin{tabular}{@{}l}{\shortstack[c]{SAMP-NUM\\SAMP-NUM-FFT\\SAMP-SIZE}} \\end{tabular}"
        ]
        # cols_format = 'l' + 'c' * (len(cols))
        cols_format = "p{1.3cm}>{\centering\\arraybackslash}p{1.5cm}>{\centering\\arraybackslash}" \
                      "p{1.5cm}>{\centering\\arraybackslash}" \
                      "p{1.cm}>{\centering\\arraybackslash}" \
                      "p{2.5cm}"
        caption = f"Feature dimensions"
        q = 0.9
    else:
        pass

    if fig_flg == 'rest':
        caption += '_rest'
    if gs:
        value_lst = value_lst[17:]  # the start of of best and default results
        caption = caption.replace('xxx', 'best')
        label = "_".join(caption.split()).replace(':', '').replace('/', '_')
    else:
        value_lst = value_lst[1:16]
        caption = caption.replace('xxx', 'default')
        label = "_".join(caption.split()).replace(':', '').replace('/', '_')

    if latex == 'figure':
        data = {'data': [], 'test_size': []}
        all_needed_data = {'data': [], 'test_size': []}
        for j, v in enumerate(value_lst):
            if "sf:True-q_flow" not in str(v[0]):  # find the first line of value
                continue

            data_flg = False
            # if "UNB/CICIDS_2017/pc_192.168.10.5" in str(v[1]) \
            #         or "UNB/CICIDS_2017/pc_192.168.10.14" in str(v[1]) \
            #         or 'CTU/IOT_2017/pc_10.0.2.15' in str(v[1]) \
            #         or 'MAWI/WIDE_2019/pc_202.171.168.50' in str(v[1]) \
            #         or 'MAWI/WIDE_2020/pc_203.78.7.165'in str(v[1]) \
            #         or 'MAWI/WIDE_2020/pc_202.119.210.242' in str(v[1]) \
            #         or 'UCHI/IOT_2019/smtv_10.42.0.1' in str(v[1]) \
            #         or 'UCHI/IOT_2019/sfrig_192.168.143.43' in str(v[1]) \
            #         or 'UCHI/IOT_2019/bstch_192.168.143.48' in str(v[1]):

            if "UNB/CICIDS_2017/pc_192.168.10.5" in str(v[1]) \
                    or "UNB/CICIDS_2017/pc_192.168.10.14" in str(v[1]) \
                    or 'CTU/IOT_2017/pc_10.0.2.15' in str(v[1]) \
                    or 'MAWI/WIDE_2019/pc_202.171.168.50' in str(v[1]) \
                    or 'UCHI/IOT_2019/smtv_10.42.0.1' in str(v[1]) \
                    or 'UCHI/IOT_2019/sfrig_192.168.143.43' in str(v[1]) \
                    or 'UCHI/IOT_2019/bstch_192.168.143.48' in str(v[1]):
                data_flg = 'default'
                # test_size=int(test_size.split()[0].split(':')[1])
                test_size = [t.split(':')[1] for t in str(v[3]).split()]

            elif "UNB/CICIDS_2017/pc_192.168.10.8" in str(v[1]) \
                    or "UNB/CICIDS_2017/pc_192.168.10.9" in str(v[1]) \
                    or "UNB/CICIDS_2017/pc_192.168.10.15" in str(v[1]) \
                    or 'UCHI/IOT_2019/ghome_192.168.143.20' in str(v[1]) \
                    or 'UCHI/IOT_2019/scam_192.168.143.42' in str(v[1]):
                data_flg = 'rest'
                test_size = [t.split(':')[1] for t in str(v[3]).split()]

            num_feats = 13

            show_needed_repres = ["STATS", "SIZE", "IAT", "IAT+SIZE", "SAMP-NUM", "SAMP-SIZE"]  # all without header
            # colors = ['m', 'green', 'darkgreen', 'red', 'darkred']
            colors = seaborn_palette(tab_type, fig_type='raw')
            value_type = 'best+min'
            v_arr = [process_value(v[10]), process_value(v[5]), process_value(v[4]), process_value(v[6]),
                     process_samp_value(v[11], value_type=value_type),
                     process_samp_value(v[12], value_type=value_type)]
            if data_flg == fig_flg:
                # data_flg = False
                tmp_arr = []
                for v_tmp in v_arr:
                    if len(v_tmp) > 4:
                        v_t = v_tmp.split('-')
                        tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                    else:
                        tmp_arr.append(v_tmp)
                all_needed_data['data'].append(v_arr)
                all_needed_data['test_size'].append(test_size)
            v_tmp = change_line(v_arr, k=2, value_type=value_type)

            if tab_type.upper() == 'basic_representation'.upper():
                representations = ["STATS", "IAT", "IAT-FFT", "SAMP-NUM", "SAMP-NUM-FFT", "SAMP-SIZE",
                                   "SAMP-SIZE-FFT"]  # all without header
                # colors = ['m', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[4]), process_value(v[7]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[14], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[15], value_type=value_type)
                         ]
                if data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'effect_size'.upper():
                representations = ["STATS", "SIZE", "IAT", "IAT+SIZE", "SAMP-NUM", "SAMP-SIZE"]  # all without header
                # colors = ['m', 'blue', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[5]), process_value(v[4]), process_value(v[6]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type)]
                if data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)  # it will change v_arr values

            elif tab_type.upper() == 'effect_header'.upper():
                representations = ["STATS (wo. header)", "STATS (w. header)", "IAT+SIZE (wo. header)",
                                   "IAT+SIZE (w. header)", "SAMP-SIZE (wo. header)",
                                   "SAMP-SIZE (w. header)"]
                # colors = ['m', 'purple', 'green', 'darkgreen', 'red', 'darkred']
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min'
                v_arr = [process_value(v[10]), process_value(v[10 + num_feats + 1]),
                         process_value(v[6]), process_value(v[6 + num_feats + 1]),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type),
                         ]
                if data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_all_without_header'.upper():
                representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
                # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min'
                v_arr = [process_value(v[4]), process_value(v[7]), process_value(v[5]), process_value(v[8]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type)]
                if data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_all_with_header'.upper():
                representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
                # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min'
                v_arr = [process_value(v[4 + num_feats + 1]), process_value(v[7 + num_feats + 1]),
                         process_value(v[5 + num_feats + 1]),
                         process_value(v[8 + num_feats + 1]),
                         process_samp_value(v[11 + num_feats + 1], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
                if data_flg == fig_flg:
                    # data_flg = False    # value is negative
                    tmp_arr = []
                    try:
                        for v_tmp in v_arr:
                            if len(v_tmp) > 4:
                                v_t = v_tmp.split('-')
                                tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                            else:
                                tmp_arr.append(v_tmp)
                    except Exception as e:
                        print(f'Error: {e}, {v_arr}')
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_samp'.upper():
                representations = ["SAMP-NUM (wo. header)", "SAMP-NUM (w. header)",
                                   "SAMP-SIZE (wo. header)", "SAMP-SIZE (w. header)"]
                # colors = ['tab:brown', 'tab:green', 'm', 'c', 'b', 'r']
                # colors = ['red', 'darkred', 'blue', 'darkblue']  # 'magenta', 'darkmagenta'
                colors = seaborn_palette(tab_type, fig_type='raw')
                value_type = 'best+min+avg'

                v_arr = [process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[11 + num_feats + 1], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
                if data_flg == fig_flg:
                    # data_flg = False
                    tmp_arr = []
                    for v_tmp in v_arr:
                        if len(v_tmp) > 4:
                            v_t = v_tmp.split('-')
                            tmp_arr.append((v_t[0].split('(')[1], v_t[1].split(')')[0]))
                        else:
                            tmp_arr.append(v_tmp)
                    data['data'].append(v_arr)
                    data['test_size'].append(test_size)
                v = change_line(v_arr, k=2, value_type=value_type)
            elif tab_type.upper() == 'feature_dimensions'.upper():
                # # representations = ["IAT/IAT-FFT","SIZE/SIZE-FFT","STATS", "SAMP-NUM/SAMP-NUM-FFT/SAMP-SIZE"]
                # representations = ["IAT"]
                # colors = ['red', 'darkred', 'blue', 'darkblue']  # 'magenta', 'darkmagenta'
                # value_type = 'wo. head'
                # if value_type == 'wo. head':
                #     v_arr = [get_dimension_from_cell(v[4]), get_dimension_from_cell(v[5]),
                #              get_dimension_from_cell(v[10]),
                #              get_dimension_from_cell(v[11], value_type=value_type)]
                # elif value_type == 'w. head':
                #     v_arr = [get_dimension_from_cell(v[4 + num_feats + 1]),
                #              get_dimension_from_cell(v[5 + num_feats + 1]),
                #              get_dimension_from_cell(v[10 + num_feats + 1]),
                #              get_dimension_from_cell(v[11 + num_feats + 1], value_type=value_type)
                #              ]
                # v = '&'.join(v_arr)
                print(f'{tab_type} does not need to show figure.')
                return -1
            else:
                v = change_line(v_arr, k=2, value_type=value_type)

        # print(j, tab_latex[s], v)
        if fig_flg == 'default':
            show_datasets = ['UNB(PC1)', 'UNB(PC4)', 'CTU', 'MAWI', 'TV&RT', 'SFrig', 'BSTch']
            # try:
            #     plot_bar(data['data'], datasets=show_datasets,
            #              repres=representations, colors=colors,
            #              output_file=os.path.dirname(output_file) + f"/{label.replace('/', '_')}.pdf")
            #     # plot_bar_difference_seaborn(data, datasets=['UNB(PC1)','UNB(PC4)', 'CTU', 'MAWI', 'TV&RT',  'SFrig', 'BSTch'],
            #     #          representations=representations, colors=colors,
            #     #          output_file=f"../output_data/{label.replace('/','_')}.pdf")
            # except Exception as e:
            #     traceback.print_exc()
        elif fig_flg == 'rest':
            show_datasets = ['UNB(PC2)', 'UNB(PC3)', 'UNB(PC5)', 'GHom', 'SCam']
            # plot_bar(data['data'], datasets=show_datasets,
            #          repres=representations, colors=colors,
            #          output_file=os.path.dirname(output_file) + f"/{label.replace('/', '_')}.pdf")
        output_file = f"{label}.pdf"  # for latex
        figure_latex = figure_temple(name=tab_type, caption=caption, cols_format=cols_format, label=label,
                                     cols=cols, output_file=output_file)
        # write to txt
        for _s, _line in enumerate(figure_latex):
            f.write(_line + '\n')
        f.write('\n\n')

        return output_file, label, caption, data, show_datasets, representations, colors, all_needed_data, show_needed_repres

    elif latex == 'table':
        tab_latex = tab_temple(name=tab_type, caption=caption, cols_format=cols_format, label=label, cols=cols)
        for s, line in enumerate(tab_latex):
            if "\midrule" in line:
                break
            f.write(line + '\n')

        data = []
        for j, v in enumerate(value_lst):
            if "sf:True-q_flow" not in str(v[0]):  # find the first line of value
                continue

            num_feats = 13
            if tab_type.upper() == 'basic_representation'.upper():
                representations = ["IAT", "IAT-FFT", "STATS", "SAMP-NUM", "SAMP-NUM-FFT"]  # all without header
                value_type = 'best+min'
                v_arr = [process_value(v[4]), process_value(v[7]), process_value(v[10]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[14], value_type=value_type)]
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'effect_size'.upper():
                representations = ["IAT", "SIZE", "IAT+SIZE", "STATS", "SAMP-NUM", "SAMP-SIZE"]  # all without header
                value_type = 'best+min'
                v_arr = [process_value(v[4]), process_value(v[5]), process_value(v[6]), process_value(v[10]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type)]
                v = change_line(v_arr, k=2, value_type=value_type)  # it will change v_arr values

            elif tab_type.upper() == 'effect_header'.upper():
                representations = ["IAT+SIZE (wo. header)", "IAT+SIZE (w. header)", "SAMP-SIZE (wo. header)",
                                   "SAMP-SIZE (w. header)", "STATS (wo. header)", "STATS (w. header)"]
                value_type = 'best+min'
                v_arr = [process_value(v[6]), process_value(v[6 + num_feats + 1]),
                         process_value(v[10]), process_value(v[10 + num_feats + 1]),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type)
                         ]
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_all_without_header'.upper():
                representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
                value_type = 'best+min'
                v_arr = [process_value(v[4]), process_value(v[7]), process_value(v[5]), process_value(v[8]),
                         process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type)]
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_all_with_header'.upper():
                representations = ["IAT", "IAT-FFT", "SIZE", "SIZE-FFT", "SAMP-NUM", "SAMP-SIZE"]
                value_type = 'best+min'
                v_arr = [process_value(v[4 + num_feats + 1]), process_value(v[7 + num_feats + 1]),
                         process_value(v[5 + num_feats + 1]),
                         process_value(v[8 + num_feats + 1]),
                         process_samp_value(v[11 + num_feats + 1], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
                v = change_line(v_arr, k=2, value_type=value_type)

            elif tab_type.upper() == 'appd_samp'.upper():
                representations = ["SAMP-NUM (wo. header)", "SAMP-NUM (w. header)",
                                   "SAMP-SIZE (wo. header)", "SAMP-SIZE\\(w. header)"]
                value_type = 'best+min+avg'
                v_arr = [process_samp_value(v[11], value_type=value_type),
                         process_samp_value(v[11 + num_feats + 1], value_type=value_type),
                         process_samp_value(v[12], value_type=value_type),
                         process_samp_value(v[12 + num_feats + 1], value_type=value_type)]
                v = change_line(v_arr, k=2, value_type=value_type)
            elif tab_type.upper() == 'feature_dimensions'.upper():
                # representations = ["IAT/IAT-FFT","SIZE/SIZE-FFT","STATS", "SAMP-NUM/SAMP-NUM-FFT/SAMP-SIZE"]
                value_type = 'wo. head'
                if value_type == 'wo. head':
                    v_arr = [get_dimension_from_cell(v[4]), get_dimension_from_cell(v[5]),
                             get_dimension_from_cell(v[10]),
                             get_dimension_from_cell(v[11], value_type=value_type)]
                elif value_type == 'w. head':
                    v_arr = [get_dimension_from_cell(v[4 + num_feats + 1]),
                             get_dimension_from_cell(v[5 + num_feats + 1]),
                             get_dimension_from_cell(v[10 + num_feats + 1]),
                             get_dimension_from_cell(v[11 + num_feats + 1], value_type=value_type)
                             ]
                v = '&'.join(v_arr)
            else:
                v = change_line(v_arr, k=2, value_type=value_type)
            # print(j, tab_latex[s], v)
            for _s, _line in enumerate(tab_latex[s:]):
                if "\midrule" in _line:
                    f.write(_line + '\n')
                    continue
                line = _line
                break
            line = line.replace('{v}', v)
            f.write(line + '\n')
            s += (_s + 1)

        for _, line in enumerate(tab_latex[s:]):
            f.write(line + '\n')

        f.write('\n\n')


def xlsx2latex_tab(input_file='', output_file='', caption='', tab_type='basic_representation', num_feat=4, verbose=1):
    """

    Parameters
    ----------
    input_file
    tab_latex
    caption
    do_header
    num_feat
    verbose

    Returns
    -------

    """

    values_dict = parse_xlsx(input_file)
    if output_file == '':
        output_file = input_file + "-" + tab_type + '-latex.txt'
    print(output_file)
    with open(output_file, 'w') as f:
        for i, (key, value_lst) in enumerate(values_dict.items()):
            print(i, key)
            if key.upper() == 'Readme'.upper():
                continue
            if key.upper() not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
                continue

            # best parameters
            append2file(f, tab_type, detector_name=key, gs=True, value_lst=value_lst[18:],
                        latex='table')  # start from the 18th row

            # default parameters
            append2file(f, tab_type, detector_name=key, gs=False, value_lst=value_lst[:18], latex='table')

    return output_file


def write_combined_reuslts_to_table(f='', tab_f='', all_neeed_f='', data={}, show_detectors='', show_datasets='',
                                    tab_type='',
                                    show_repres='',
                                    gs='', fig_flg='default', caption='', appendix_flg=False):
    s_data = {}
    for j, detector_name in enumerate(show_detectors):
        sub_dataset = []
        new_colors = []
        for ind, dataset_name in enumerate(show_datasets):
            if tab_type.upper() == "basic_representation".upper():
                # features = ['STATS', 'IAT', 'IAT-FFT', 'SAMP-NUM', 'SAMP-NUM-FFT']
                f_dict = dict(zip(show_repres, [i for i in range(len(show_repres))]))
                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'STATS'
                B = 'IAT'
                C = 'SAMP-NUM'
                A_v = [data[detector_name][dataset_name][gs][-1][f_dict[A]], 0][0]
                B_v = [data[detector_name][dataset_name][gs][-1][f_dict[B]], 0][0]
                C_v = data[detector_name][dataset_name][gs][-1][f_dict[C]]
                C_vt = C_v.split('-')
                # C_v = C_vt[0].split('(')[1] + '(' + C_vt[1].split(')')[1].strip() +')'
                C_v = C_vt[0].split('(')[1].strip()
                sub_dataset.append([A_v, B_v, C_v])
                caption = 'Results with xxx parameters on basic representations'
                # show_repres = [A, B, C]
                feature_cols = [A, B, C]
                cols_format = 'C|' * len(feature_cols)  # C| for tabluary

            elif tab_type.upper() == "effect_size".upper():
                # features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP+SIZE']
                f_dict = dict(zip(show_repres, [i for i in range(len(show_repres))]))

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'STATS'
                B = 'SIZE'
                C = 'IAT'
                D = 'SAMP-NUM'
                A_v = [data[detector_name][dataset_name][gs][-1][f_dict[A]], 0][0]
                B_v = [data[detector_name][dataset_name][gs][-1][f_dict[B]], 0][0]
                C_v = [data[detector_name][dataset_name][gs][-1][f_dict[C]], 0][0]
                D_v = data[detector_name][dataset_name][gs][-1][f_dict[D]]
                D_vt = D_v.split('-')
                # D_v = D_vt[0].split('(')[1] + '(' + D_vt[1].split(')')[1].strip() + ')'
                D_v = D_vt[0].split('(')[1].strip()
                sub_dataset.append([A_v, B_v, C_v, D_v])
                caption = 'Effect of Size: Results with xxx parameters'
                # show_repres = [C, D]
                feature_cols = [C, D]
                cols_format = 'C|' * len(feature_cols)

            elif tab_type.upper() == "effect_header".upper():
                # features = ['STATS (wo. header)', 'STATS (w. header)', 'IAT+SIZE (wo. header)', 'IAT+SIZE (w. header)',
                #             'SAMP-SIZE (wo. header)', 'SAMP-SIZE (w. header)']
                f_dict = dict(zip(show_repres, [i for i in range(len(show_repres))]))
                A = 'STATS (wo. header)'
                B = 'IAT+SIZE (wo. header)'
                C = 'SAMP-SIZE (wo. header)'
                A_v = [data[detector_name][dataset_name][gs][-1][f_dict[A]], 0][0]
                B_v = [data[detector_name][dataset_name][gs][-1][f_dict[B]], 0][0]
                C_v = data[detector_name][dataset_name][gs][-1][f_dict[C]]
                C_vt = C_v.split('-')
                # C_v = C_vt[0].split('(')[1] + '(' + C_vt[1].split(')')[1].strip() + ')'
                C_v = C_vt[0].split('(')[1].strip()
                sub_dataset.append([A_v, B_v, C_v])
                # caption = 'Effect of Packet Header: Results with xxx parameters'
                caption = 'Results with xxx parameters on representations without packet header'
                # show_repres = [B, C]
                # feature_cols = ['STATS \\\(wo. header)', 'IAT+SIZE \\\(wo. header)', 'SAMP-SIZE \\\(wo. header)']
                feature_cols = ['IAT+SIZE', 'SAMP-SIZE']
                cols_format = 'C|' * len(feature_cols)

            elif tab_type.upper() == "all_needed_data".upper():
                # features = ['STATS','SIZE', 'IAT', 'IAT+SIZE','SAMP-NUM',
                #             'SAMP-SIZE']
                f_dict = dict(zip(show_repres, [i for i in range(len(show_repres))]))
                A = 'STATS'
                B = 'SIZE'
                C = 'IAT'
                D = 'IAT+SIZE'
                E = 'SAMP-NUM'
                F = 'SAMP-SIZE'
                A_v = [data[detector_name][dataset_name][gs][-1][f_dict[A]], 0][0]
                B_v = [data[detector_name][dataset_name][gs][-1][f_dict[B]], 0][0]
                C_v = [data[detector_name][dataset_name][gs][-1][f_dict[C]], 0][0]
                D_v = [data[detector_name][dataset_name][gs][-1][f_dict[D]], 0][0]
                E_v = data[detector_name][dataset_name][gs][-1][f_dict[E]]
                E_vt = E_v.split('-')
                E_v = E_vt[0].split('(')[1].strip()
                F_v = data[detector_name][dataset_name][gs][-1][f_dict[F]]
                F_vt = F_v.split('-')
                F_v = F_vt[0].split('(')[1].strip()
                sub_dataset.append([A_v, B_v, C_v, D_v, E_v, F_v])
                # caption = 'Effect of Packet Header: Results with xxx parameters'
                caption = 'Results with xxx parameters on representations without packet header'
                # show_repres = [A, B, C, D, E, F]
                # feature_cols = ['STATS \\\(wo. header)', 'IAT+SIZE \\\(wo. header)', 'SAMP-SIZE \\\(wo. header)']
                feature_cols = show_repres
                cols_format = 'C|' * len(feature_cols)
            else:
                caption = tab_type
                cols_format = ''

        s_data[detector_name] = sub_dataset

    det_str = ''
    for i, v in enumerate(show_detectors):
        if i < len(show_detectors) - 1:
            det_str += v + ', '
        else:
            det_str += f'and {v}'
    if len(show_detectors) == 2:
        det_str = det_str.replace(',', '')
    elif len(show_detectors) == 1:
        det_str = det_str.replace('and', '')
    caption = caption.replace('Results with', f'{det_str} with')

    # write to table
    if gs:
        caption = caption.replace('xxx', 'best')
        label = '_'.join(caption.split())
    else:
        caption = caption.replace('xxx', 'default')
        label = '_'.join(caption.split())
    if fig_flg == 'rest':
        caption = caption + ' (5 other datasets)'
        label = label + f'_{fig_flg}'

    label = label.replace(':', '').replace(',', '')

    if tab_type == 'all_needed_data':
        cols_format = '|@{}C|C|' + cols_format  # @{} with indent in the cell
        if gs:
            para = 'best parameters'
        else:
            para = 'default parameters'
        caption = f"{det_str} with {para}"
        label = '_'.join(caption.split()).replace(',', '')
        if fig_flg == 'rest':
            caption = caption + ' (5 other datasets)'
            label = label + f'_{fig_flg}'

        tab_latex = comb_all_needed_data_tab_temple(name=tab_type, caption=caption, cols_format=cols_format,
                                                    label=label,
                                                    detector_cols=show_detectors, feature_cols=feature_cols,
                                                    fig_flg=fig_flg, appendix_flg=appendix_flg)

        # write begin
        for s, line in enumerate(tab_latex):
            if "\midrule" in line or "\cmidrule" in line:
                break
            all_neeed_f.write(line + '\n')

        for j, detector_name in enumerate(show_detectors):
            line = []
            for ind, dataset_name in enumerate(show_datasets):
                line = s_data[detector_name][ind]
                v_str = " & ".join([str(v) for v in line])

                # print(j, tab_latex[s], v)
                for _s, _line in enumerate(tab_latex[s:]):
                    if "\midrule" in _line or "\cmidrule" in _line:
                        all_neeed_f.write(_line + '\n')
                        continue
                    line = _line
                    break
                s += (_s + 1)
                print(j, detector_name, ind, dataset_name, line, v_str, flush=True)
                line = line.replace('{v}', v_str)
                all_neeed_f.write(line + '\n')

        # write end
        for _, line in enumerate(tab_latex[s:]):
            all_neeed_f.write(line + '\n')

        all_neeed_f.write('\n\n')

    else:
        cols_format = '|@{}L|' + cols_format * len(show_detectors)  # @{} with indent in the cell
        tab_latex = comb_tab_temple(name=tab_type, caption=caption, cols_format=cols_format, label=label,
                                    detector_cols=show_detectors, feature_cols=feature_cols, fig_flg=fig_flg)
        # write begin
        for s, line in enumerate(tab_latex):
            if "\midrule" in line:
                break
            tab_f.write(line + '\n')
            f.write(line + '\n')

        for ind, dataset_name in enumerate(show_datasets):
            line = []
            for j, detector_name in enumerate(show_detectors):
                line += s_data[detector_name][ind]
            v_str = " & ".join([str(v) for v in line])

            # print(j, tab_latex[s], v)
            for _s, _line in enumerate(tab_latex[s:]):
                if "\midrule" in _line:
                    tab_f.write(_line + '\n')
                    f.write(_line + '\n')
                    continue
                line = _line
                break
            s += (_s + 1)
            line = line.replace('{v}', v_str)
            tab_f.write(line + '\n')
            f.write(line + '\n')

        # write end
        for _, line in enumerate(tab_latex[s:]):
            tab_f.write(line + '\n')
            f.write(line + '\n')

        tab_f.write(line + '\n')
        f.write('\n\n')

    return


def xlsx2latex_figure(input_file='', output_file='output_data', caption='', tab_type='basic_representation',
                      fig_flg='default', appendix=True, gs=True, f='',
                      num_feat=4, verbose=1):
    """

    Parameters
    ----------
    input_file
    tab_latex
    caption
    do_header
    num_feat
    verbose

    Returns
    -------

    """

    values_dict = parse_xlsx(input_file)
    if output_file == '':
        output_file = input_file + "-" + tab_type + '-latex-figures.txt'
    print(output_file)

    with open(output_file, 'w') as out_f:
        datasets = OrderedDict()
        all_needed_datasets = OrderedDict()
        for i, (key, value_lst) in enumerate(values_dict.items()):
            print(i, key)
            key = key.upper()
            if key == 'Readme'.upper():
                continue
            if key not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
                # if key.upper() not in ['PCA', 'AE']:
                continue

            fig1, label1, caption1, data1, show_dataset1, show_repres1, colors1, all_needed_data, show_needed_repres \
                = append2file(out_f, tab_type,
                              detector_name=key, gs=gs,
                              value_lst=value_lst,
                              latex='figure',
                              fig_flg=fig_flg,
                              output_file=output_file)  # start from the 18th row
            if appendix:
                fig2, label2, caption2, data2, show_dataset2, show_repres2, colors2, all_needed_data2, show_needed_repres2 \
                    = append2file(out_f, tab_type,
                                  detector_name=key, gs=gs,
                                  value_lst=value_lst,
                                  latex='figure',
                                  fig_flg='rest',
                                  output_file=output_file)  # start from the 18th row
                data1['data'] += data2['data']
                data1['test_size'] += data2['test_size']
                show_dataset1 += show_dataset2
                all_needed_data['data'] += all_needed_data2['data']
                all_needed_data['test_size'] += all_needed_data2['test_size']

                # data1: UNB(PC1), UNB(PC4), CTU, MAWI, TV&RT,SFrig, and BSTCH
                # data2: UNB(PC2), UNB(PC3), UNB(PC5),  GHom, SCam
                # UNB(PC1), UNB(PC2), UNB(PC3), UNB(PC4), UNB(PC5), CTU, MAWI, TV&RT, GHom, SCam, SFrig, and BSTCH
                def change_order(data):
                    # change: [UNB(PC1), UNB(PC4), CTU, MAWI, TV&RT,SFrig, BSTCH, UNB(PC2), UNB(PC3), UNB(PC5), GHom,SCam]
                    # to [UNB(PC1), UNB(PC2), UNB(PC3), UNB(PC4), UNB(PC5), CTU, MAWI, TV&RT, GHom, SCam, SFrig, BSTCH]
                    _data = [data[0], data[7], data[8], data[1], data[9], data[2], data[3], data[4], data[10], data[11],
                             data[5], data[6]]
                    return _data

                data1['data'] = change_order(data1['data'])
                data1['test_size'] = change_order(data1['test_size'])
                show_dataset1 = change_order(show_dataset1)
                all_needed_data['data'] = change_order(all_needed_data['data'])
                all_needed_data['test_size'] = change_order(all_needed_data['test_size'])

            # for d_name, d in zip(show_dataset1, data1):
            #     if d_name not in datasets.keys():
            #         datasets[d_name] = {}
            #     if key.upper() not in datasets[d_name].keys():
            #         datasets[d_name][key.upper()] = {}
            #     if gs not in datasets[d_name][key.upper()].keys():
            #         datasets[d_name][key.upper()][gs] = {}
            #     datasets[d_name][key.upper()][gs] = (fig1, label1, caption1, d)

            if key not in datasets.keys():
                datasets[key] = {}
                for d_name, d, ts in zip(show_dataset1, data1['data'], data1['test_size']):
                    if d_name not in datasets[key].keys():
                        datasets[key][d_name] = {'test_size': ts}
                    if gs not in datasets[key][d_name].keys():
                        datasets[key][d_name][gs] = {}
                    datasets[key][d_name][gs] = (fig1, label1, caption1, d)
            else:
                for d_name, d, ts in zip(show_dataset1, data1['data'], data1['test_size']):
                    if d_name not in datasets[key].keys():
                        datasets[key][d_name] = {'test_size': ts}
                    if gs not in datasets[key][d_name].keys():
                        datasets[key][d_name][gs] = {}
                    datasets[key][d_name][gs] = (fig1, label1, caption1, d)

            if key not in all_needed_datasets.keys():
                all_needed_datasets[key] = {}
                for d_name, d, ts in zip(show_dataset1, all_needed_data['data'], all_needed_data['test_size']):
                    if d_name not in all_needed_datasets[key].keys():
                        all_needed_datasets[key][d_name] = {'test_size': ts}
                    if gs not in all_needed_datasets[key][d_name].keys():
                        all_needed_datasets[key][d_name][gs] = {}
                    all_needed_datasets[key][d_name][gs] = (fig1, label1, caption1, d)
            else:
                for d_name, d, ts in zip(show_dataset1, all_needed_data['data'], all_needed_data['test_size']):
                    if d_name not in all_needed_datasets[key].keys():
                        all_needed_datasets[key][d_name] = {'test_size': ts}
                    if gs not in all_needed_datasets[key][d_name].keys():
                        all_needed_datasets[key][d_name][gs] = {}
                    all_needed_datasets[key][d_name][gs] = (fig1, label1, caption1, d)

            # # default parameters
            # gs = False
            # fig2, label2, caption2, data2, show_dataset2, show_repres2, colors2 = append2file(out_f, tab_type,
            #                                                                                   detector_name=key, gs=gs,
            #                                                                                   value_lst=value_lst[:18],
            #                                                                                   latex='figure',
            #                                                                                   fig_flg=fig_flg,
            #                                                                                   output_file=output_file)
            # # for d_name, d in zip(show_dataset2, data2):
            # #     if d_name not in datasets.keys():
            # #         datasets[d_name] = {}
            # #     if key.upper() not in datasets[d_name].keys():
            # #         datasets[d_name][key.upper()] = {}
            # #     if gs not in datasets[d_name][key.upper()].keys():
            # #         datasets[d_name][key.upper()][gs] = {}
            # #     datasets[d_name][key.upper()][gs] = (fig2, label2, caption2, d)
            #
            # if key not in datasets.keys():
            #     datasets[key] = {}
            #     for d_name, d, ts in zip(show_dataset2, data2['data'], data2['test_size']):
            #         if d_name not in datasets[key].keys():
            #             datasets[key][d_name] = {'test_size': ts}
            #         if gs not in datasets[key][d_name].keys():
            #             datasets[key][d_name][gs] = {}
            #         datasets[key][d_name][gs] = (fig2, label2, caption2, d)
            # else:
            #     for d_name, d, ts in zip(show_dataset2, data2['data'], data2['test_size']):
            #         if d_name not in datasets[key].keys():
            #             datasets[key][d_name] = {'test_size': ts}
            #         if gs not in datasets[key][d_name].keys():
            #             datasets[key][d_name][gs] = {}
            #         datasets[key][d_name][gs] = (fig2, label2, caption2, d)
            #
            # caption = caption1.replace('best', 'best and default')
            # label = '_'.join(caption.split())
            # comb_figures_latex = comb_figures_temple(name=tab_type, detector_name=key,
            #                                          caption=caption, label=label,
            #                                          fig1=fig1, fig1_label=label1,
            #                                          fig2=fig2, fig2_label=label2)
            # # write to txt
            # for _s, _line in enumerate(comb_figures_latex):
            #     f.write(_line + '\n')
            # f.write('\n\n')
        # print(label1, label2)

        # show_detectors_lst = [['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]
        if not appendix:
            show_detectors_lst = [['OCSVM', 'IF', 'AE', 'KDE']]
        else:
            show_detectors_lst = [['OCSVM', 'IF', 'AE', 'KDE'], ['GMM', 'PCA']]
            # show_detectors_lst = [['GMM', 'PCA']]
        # # output_file_diff = output_file + f"-{fig_flg}-diff-gs_{gs}-latex-merged_tab.txt"
        # with open(output_file_diff, 'w') as all_neeed_f:
        #     for show_detectors in show_detectors_lst:
        #         print(f'fig_flg:{fig_flg}, show_detectors: {show_detectors}, gs={gs}')
        #         # name = "_".join(show_detectors)
        #         write_combined_reuslts_to_table(all_neeed_f=all_neeed_f, data=all_needed_datasets, show_detectors=show_detectors,
        #                                         show_datasets=show_dataset1, tab_type='all_needed_data',
        #                                         show_repres=show_needed_repres, fig_flg=fig_flg, caption=caption1,
        #                                         gs=gs)

        # output_file_diff = output_file + f"-{fig_flg}-diff-gs_{gs}-latex-tab.txt"
        # output_file_fig = output_file + f"-{fig_flg}_diff-gs{gs}-latex-fig.txt"  # for latex
        output_file_diff = output_file + f"-{fig_flg}-diff-gs_{gs}-latex-tab-appendix.txt"
        output_file_fig = output_file + f"-{fig_flg}_diff-gs{gs}-latex-fig-appendix.txt"  # for latex
        with open(output_file_diff, 'w') as tab_f:
            with open(output_file_fig, 'w') as fig_f:
                for show_detectors in show_detectors_lst:
                    print(f'fig_flg:{fig_flg}, show_detectors: {show_detectors}, gs={gs}')
                    # name = "_".join(show_detectors)
                    write_combined_reuslts_to_table(all_neeed_f=f, data=all_needed_datasets,
                                                    show_detectors=show_detectors,
                                                    show_datasets=show_dataset1, tab_type='all_needed_data',
                                                    show_repres=show_needed_repres, fig_flg=fig_flg, caption=caption1,
                                                    gs=gs, appendix_flg=appendix)

                    if not appendix:
                        write_combined_reuslts_to_table(f=f, tab_f=tab_f, data=datasets, show_detectors=show_detectors,
                                                        show_datasets=show_dataset1, tab_type=tab_type,
                                                        show_repres=show_repres1, fig_flg=fig_flg, caption=caption1,
                                                        gs=gs, appendix_flg=appendix)
                    caption = caption1
                    det_str = ''
                    for i_v, v in enumerate(show_detectors):
                        if i_v < len(show_detectors) - 1:
                            det_str += f'{v}, '
                        else:
                            det_str += f'and {v}'
                    if len(show_detectors) == 1:
                        det_str = det_str.replace('and ', '')
                    elif len(show_detectors) == 2:
                        det_str = det_str.replace(',', '')
                    else:
                        pass
                    if tab_type.upper() == 'basic_representation'.upper():
                        tmp = caption.split()
                        tmp[0] = det_str
                    elif tab_type.upper() == 'effect_size'.upper():
                        tmp = caption.split()
                        tmp[3] = det_str
                    elif tab_type.upper() == 'effect_header'.upper():
                        tmp = caption.split()
                        tmp[4] = det_str
                    caption = " ".join(tmp)
                    label = "_".join(tmp).replace(':', '').replace(' ', '_').replace(',', '').replace('and_', '')

                    if appendix:
                        output_file_diff1 = os.path.dirname(output_file_diff) + f"/{label}_appendix_{appendix}_diff.pdf"
                    else:
                        output_file_diff1 = os.path.dirname(output_file_diff) + f"/{label}_diff.pdf"
                    new_data, new_colors = plot_bar_difference_seaborn(datasets, show_detectors=show_detectors,
                                                                       show_datasets=show_dataset1, gs=gs,
                                                                       show_repres=show_repres1, colors=colors1,
                                                                       output_file=output_file_diff1,
                                                                       tab_type=tab_type, appendix=appendix)

                    output_file_diff2 = os.path.dirname(output_file_diff) + f"/{label}_diff_on_each_dataset.pdf"
                    # plot_bar_ML_seaborn(new_data, new_colors=new_colors, show_detectors=show_detectors,
                    #                     output_file=output_file_diff2, gs=gs,
                    #                     tab_type=tab_type)

                    # write to fig.txt
                    if tab_type.upper() == 'basic_representation'.upper():
                        det_str = caption.split('diff')[0]
                        caption = f'The AUC difference: IAT vs. IAT-FFT, and SAMP-NUM vs. SAMP-NUM-FFT, generated by {det_str}'
                    elif tab_type.upper() == 'effect_size'.upper():
                        det_str = caption.split('diff')[0].split('Size:')[1]
                        caption = f'Effect of Size: the AUC difference: IAT vs. IAT+SIZE, and SAMP-NUM vs. SAMP-SIZE, generated by {det_str}'
                    elif tab_type.upper() == 'effect_header'.upper():
                        det_str = caption.split('diff')[0].split('Header:')[1]
                        caption = f'Effect of Packet Header: The AUC difference: STATS (wo. header) vs. STATS (w. header), ' \
                                  f'IAT+SIZE (wo. header) vs. IAT+SIZE (w. header), and SAMP-SIZE (wo. header) vs. SAMP-SIZE (w. header), ' \
                                  f'generated by {det_str}'
                    label = label.replace('and_', '')
                    caption = caption.replace('_rest', ' (5 other datasets)')
                    caption = ' '.join(caption.split())
                    for i_fig, fig_file in enumerate([output_file_diff1, output_file_diff2]):
                        if i_fig == 1:
                            caption += ' on each dataset'
                        # fig_file = f'{label}.pdf'
                        fig_name = os.path.basename(fig_file).replace('and_', '')
                        figure_latex = figure_temple(name='diff', caption=caption, label=fig_name.split('.pdf')[0],
                                                     output_file=fig_name)
                        # write to txt
                        for _s, _line in enumerate(figure_latex):
                            fig_f.write(_line + '\n')
                            f.write(_line + '\n')
                        fig_f.write('\n\n')
                        f.write('\n\n')
    return output_file


def get_color_position(values, k=2, reverse=True):
    try:
        arr_tmp_f = []
        for v in values:
            if '(' in str(v):
                v = v.split('(')[0]
            if '=' in str(v):
                v = v.split('=')[-1]
            arr_tmp_f.append(float(f'{float(v):.4f}'))
        sorted_arr, indices, same_values = find_top_k_values(arr_tmp_f, k=k, reverse=reverse)
        return sorted_arr, indices, same_values

    except Exception as e:
        print(f'Error: {e}')


# def label_one_cell(v, text_wrap):
#
#     cell_color=[]
#     # find 'std'
#     v_t = v.split('|')
#     st = len(v_t[0]) + 1 + len(v_t[1]) + 1 + len(v_t[2]) + 1
#     ed = st + len(v_t[3]) + 1
#     # v[st:ed] = samp_colors[color_idx]
#     # cell_color.extend(['color:red; font-weight:bold','', 'color:blue; font-weight:bold'])
#     cell_color='color:red; font-weight:bold, , color:blue; font-weight:bold'
#
#     # find max 'aucs'
#     st = len(v_t[0]) + 1 + len(v_t[1]) + 1 + len(v_t[2]) + 1 + len(v_t[3]) + 1
#     v_aucs = v_t[-1]
#     v_h = v_aucs.split('=')
#     st += len(v_h[0]) + 1
#     aucs = v_h[-1].split(')')[0].split('+')
#
#     sorted_arr, indices, same_values = get_color_position(aucs, k=1)
#     # print('Top 2: ', indices)
#     print(sorted_arr, indices, same_values)
#     for _, (v1, idx1, color_idx1) in enumerate(zip(sorted_arr, indices, same_values)):
#         ed = st + 2 * len(str(v1)) + 1
#         # v[st:ed] = samp_colors[color_idx1]
#         # cell_color.append('color:black; font-weight:bold')
#
#     return cell_color


def line_format(worksheet, workbook, values, row, begin=4, header=False, simple_flg=False):
    red_color = workbook.add_format({'color': 'red', 'bold': True})
    blue_color = workbook.add_format({'color': 'blue', 'bold': True})
    magenta = workbook.add_format({'color': 'magenta', 'bold': True})
    purple = workbook.add_format({'color': 'purple', 'bold': True})

    bold = workbook.add_format({'bold': True})
    text_wrap = workbook.add_format({'text_wrap': False})

    colors = [red_color, blue_color, magenta, purple]

    end = begin + 13 - 3 - 3
    vs = ['' if v == 'nan' or '' else str(v) for v in values[row][begin:end]]
    sorted_arr, indices, same_values = get_color_position(vs, k=2)
    for _, (v1, idx1, color_idx1) in enumerate(zip(sorted_arr, indices, same_values)):
        vs[idx1] = [' ', colors[color_idx1], str(vs[idx1])]  # [ red_color, str(vs[idx1])]. which one doesn't work.

    if header:
        provit = 18
    else:
        provit = 4
    for c in range(begin, end):
        # print(vs[c-4])
        if type(vs[c - provit]) == list:
            string_parts = vs[c - provit]
            # string_parts.append(text_wrap)
            # print(*string_parts)
            worksheet.write_rich_string(row, c, *string_parts)
        else:
            worksheet.write(row, c, vs[c - provit])

    # process the samp features
    begin = end
    end = begin + 3 + 3
    # arr_tmp=[df.iloc[i,j] for j in range(begin, end)]

    brown = workbook.add_format({'color': 'brown', 'bold': True})
    black = workbook.add_format({'color': 'black', 'bold': False})
    MAROON = workbook.add_format({'color': '#800000', 'bold': True})
    OLIVE = workbook.add_format({'color': '#808000', 'bold': True})
    TEAL = workbook.add_format({'color': '#008080', 'bold': True})

    orange = workbook.add_format({'color': 'orange', 'bold': True})
    NAVY = workbook.add_format({'color': '#000080', 'bold': True})

    colors_auc = [TEAL, orange, NAVY, MAROON, OLIVE, ]
    for c in range(begin, end):
        try:
            v_arr = values[row][c].split('|')
            if '-' in v_arr[0]:
                continue
        except Exception as e:
            print(e, c, values[row][c])
            continue
        # print(row, c, begin, end, v_arr)
        if simple_flg:
            string_parts = [black, v_arr[-1][:-1].split('=')[-1], '(' + v_arr[2] + ')']
        else:
            string_parts = [v_arr[0] + '|', v_arr[1] + '|', MAROON, v_arr[2] + '|', NAVY, v_arr[3] + '|']

            v_t = v_arr[4].split('=')
            string_parts += [v_t[0] + '=']
            v_aucs = v_t[1].split(')')[0].split('+')
            k_max = 1
            v_tmps = copy.deepcopy(v_aucs)
            sorted_arr, indices, same_values = get_color_position(v_aucs, k=k_max, reverse=True)  # find the maximum
            for _, (v1, idx1, color_idx1) in enumerate(zip(sorted_arr, indices, same_values)):
                v_tmps[idx1] = [' ', colors_auc[color_idx1], v_aucs[idx1]]

            k_min = 1
            if len(colors_auc) < (k_max + k_min):
                raise ValueError('colors_auc is not enough')
            sorted_arr, indices, same_values = get_color_position(v_aucs, k=k_min, reverse=False)  # find the minimum
            for _, (v1, idx1, color_idx1) in enumerate(zip(sorted_arr, indices, same_values)):
                v_tmps[idx1] = [' ', colors_auc[k_max + color_idx1], v_aucs[idx1]]

            for s, v in enumerate(v_tmps):
                if type(v) == list:
                    if s == len(v_tmps) - 1:
                        string_parts.extend(v)
                    else:
                        string_parts.extend(v + ['+'])
                else:  # str
                    if s == len(v_tmps) - 1:
                        string_parts.append(v)
                    else:
                        string_parts.append(v + '+')
            # vote_auc
            string_parts += ['|', OLIVE, v_arr[-1]]
            string_parts.append(text_wrap)
            # print(row, c, *string_parts)
            # worksheet.write_rich_string(row, c, *string_parts)
        worksheet.write_rich_string(row, c, *string_parts)

    return worksheet


def hightlight_excel(input_file, output_file=''):
    xls = pd.ExcelFile(input_file)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)
    import xlsxwriter
    workbook = xlsxwriter.Workbook(output_file)

    for t, sheet_name in enumerate(xls.sheet_names):
        print(t, sheet_name)
        if sheet_name.upper() not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
            continue
        worksheet = workbook.add_worksheet(sheet_name)

        df = pd.read_excel(input_file, sheet_name=sheet_name, header=None,
                           index_col=None)  # index_col=False: not use the first columns as index
        values = df.values
        rows, cols = values.shape
        for row in range(rows):

            line = str(values[row][0])
            if ('sf' in line) and ('q_flow' in line) and ('interval' in line):
                try:

                    begin = 0
                    end = begin + 4
                    vs = ['' if v == 'nan' or v == '' else str(v) for v in values[row][begin:end]]
                    worksheet.write_row(row, 0, vs)  # write a list from (row, col)

                    worksheet = line_format(worksheet, workbook, values, row, begin=4, header=False)

                    worksheet = line_format(worksheet, workbook, values, row, begin=4 + 13 + 1, header=True)

                except Exception as e:
                    print(f'Error: {e}')
            else:
                vs = ['' if str(v).strip() == 'nan' or str(v).strip() == '' else str(v) for v in values[row]]
                print(row, vs)
                worksheet.write_row(row, 0, vs)  # write a list from (row, col)

    workbook.close()

    return output_file


def highlight_cell(row):
    """
        Reference: 1) https://stackoverflow.com/questions/43596579/how-to-use-python-pandas-stylers-for-coloring-an-entire-row-based-on-a-given-col
            combined formats for cells: 2) https://stackoverflow.com/questions/58344309/pandas-conditionally-format-field-in-bold
            3) https://www.color-hex.com/color/474747
    Parameters
    ----------
    row

    Returns
    -------

    """
    # print(f'len(row): {len(row)}, {row}')

    if ('sf' not in str(row[0])) or ('q_flow' not in str(row[0])) or ('interval' not in str(row[0])):
        return [''] * len(row)

    # print(row.values)
    row_color = [''] * len(row)
    # colors = ['background-color: yellow', 'background-color: red', 'background-color: blue', 'background-color: green']
    # colors = ['color: black', 'color: red', 'color: blue', 'color: green']

    colors = ['color:red; font-weight:bold', 'color:blue; font-weight:bold', 'color:black; font-weight:bold',
              'color:#707070; font-weight:bold']
    samp_colors = ['color:red; font-weight:bold', 'color:blue; font-weight:bold', 'color:black; font-weight:bold',
                   'color:#707070; font-weight:bold']

    begin_col = 4
    end_col = 4 + 13  # 13 features
    values = row[begin_col:end_col].values
    all_flg = True
    if '-' in values:
        # row_color = [ '' for i, v in enumerate(row_color) if i >= 4 and i<11]

        # for i, v in enumerate(row_color):
        #     if i >= 4 and i < 11:
        #         row_color[i] =''
        row_color = ['' if i >= begin_col and i < end_col else v for i, v in enumerate(row_color)]
    else:
        if all_flg:
            sorted_arr, indices, same_values = get_color_position(values)
            # print('Top 2: ', indices)
            print(sorted_arr, indices, same_values)
            for i, (v, idx, color_idx) in enumerate(zip(sorted_arr, indices, same_values)):
                row_color[idx + begin_col] = colors[color_idx]
        else:
            values = row[begin_col:end_col - 3].values
            sorted_arr, indices, same_values = get_color_position(values)
            # print('Top 2: ', indices)
            print(sorted_arr, indices, same_values)
            for i, (v, idx, color_idx) in enumerate(zip(sorted_arr, indices, same_values)):
                row_color[idx + begin_col] = colors[color_idx]

            # begin_col = end_col-3
            # ### label samp_value
            # for idx, v in enumerate(row[end_col-3:end_col].values):
            #     row_color[idx + begin_col] = label_one_cell(v, samp_colors=[])

    begin_col = 18
    end_col = 18 + 13  # 13 features
    values = row[begin_col:end_col].values
    if '-' in values:
        # for i, v in enumerate(row_color):
        #     if i >= 13 and i < 20:
        #         row_color[i] =''
        # row_color = [  for i, v in enumerate(row_color) if i >= 13 and i<20]
        row_color = ['' if i >= begin_col and i < end_col else v for i, v in enumerate(row_color)]
    else:
        # sorted_arr, indices, same_values = get_color_position(values)
        # # print('Top 2: ', indices)
        # for i, (v, idx, color_idx) in enumerate(zip(sorted_arr, indices, same_values)):
        #     row_color[idx + begin_col] = colors[color_idx]

        if all_flg:
            sorted_arr, indices, same_values = get_color_position(values)
            # print('Top 2: ', indices)
            print(sorted_arr, indices, same_values)
            for i, (v, idx, color_idx) in enumerate(zip(sorted_arr, indices, same_values)):
                row_color[idx + begin_col] = colors[color_idx]
        else:
            values = row[begin_col:end_col - 3].values
            sorted_arr, indices, same_values = get_color_position(values)
            # print('Top 2: ', indices)
            print(sorted_arr, indices, same_values)
            for i, (v, idx, color_idx) in enumerate(zip(sorted_arr, indices, same_values)):
                row_color[idx + begin_col] = colors[color_idx]

    return row_color


@func_notation
def xlsx_highlight_cells(file_name, output_file='example_highlight.xlsx'):
    xls = pd.ExcelFile(file_name)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)

    with ExcelWriter(output_file) as writer:
        for i, sheet_name in enumerate(xls.sheet_names):
            print(i, sheet_name)
            if sheet_name.upper() not in ['OCSVM', 'KDE', 'GMM', 'IF', 'PCA', 'AE']:
                continue
            df = pd.read_excel(file_name, sheet_name=sheet_name, header=None,
                               index_col=None)  # index_col=False: not use the first columns as index
            # styled_df = df.style.apply(highlight_cell, axis=1, subset=[0, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19])
            styled_df = df.style.apply(highlight_cell, axis=1)
            styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        writer.save()

    return output_file


def get_diff(values_1, values_2):
    if ('sf' not in str(values_1[0])) or ('q_flow' not in str(values_1[0])) or ('interval' not in str(values_1[0])):
        return values_1

    values = []
    start_idx = 4
    for i, (v_1, v_2) in enumerate(zip(values_1, values_2)):
        if i < start_idx:
            values.append(v_1)
            continue
        if '(' in str(v_1):
            try:
                v_1 = v_1.split('(')[0]
                v_2 = v_2.split('(')[0]
                values.append(float(f'{float(v_1) - float(v_2):.4f}'))
            except Exception as e:
                print(f'i {i}, Error: {e}')
                values.append('-')
        else:
            values.append(v_1)

    return values


def diff_files(file_1, file_2, output_file='diff.xlsx', file_type='xlsx'):
    if file_type in ['xlsx']:
        xls = pd.ExcelFile(file_1)
        # Now you can list all sheets in the file
        print(f'xls.sheet_names:', xls.sheet_names)
        with ExcelWriter(output_file) as writer:
            for i, sheet_name in enumerate(xls.sheet_names):
                print(i, sheet_name)
                df_1 = pd.read_excel(file_1, sheet_name=sheet_name, header=None,
                                     index_col=None)  # index_col=False: not use the first columns as index

                df_2 = pd.read_excel(file_2, sheet_name=sheet_name, header=None,
                                     index_col=None)  # index_col=False: not use the first columns as index

                values = []
                for j, (v_1, v_2) in enumerate(list(zip(df_1.values, df_2.values))):
                    values.append(get_diff(v_1, v_2))
                # Generate dataframe from list and write to xlsx.
                pd.DataFrame(values).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                # styled_df = df_1.style.apply(highlight_cell, axis=1)
                # styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            writer.save()

    return output_file


def merge_xlsx(input_files=[], output_file='merged.xlsx'):
    workbook = xlsxwriter.Workbook(output_file)
    for t, (sheet_name, input_file) in enumerate(input_files):
        print(t, sheet_name, input_file)
        worksheet = workbook.add_worksheet(sheet_name)
        if not os.path.exists(input_file):
            # for i in range(rows):
            #     worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
            #                                    list(values[i])])  # write a list from (row, col)
            pass
        else:
            df = pd.read_excel(input_file, header=None, index_col=None)
            values = df.values
            rows, cols = values.shape
            # add column index
            # worksheet.write_row(0, 0, [str(i) for i in range(cols)])
            for i in range(rows):
                worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
                                           list(values[i])])  # write a list from (row, col)
            # worksheet.write(df.values)
    workbook.close()

    return output_file


def clean_xlsx(input_file, output_file='clean.xlsx'):
    xls = pd.ExcelFile(input_file)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)

    def process_line(v_arr):
        if 'sf:True-q_flow_dur:' in str(v_arr[1]):
            for i, v_t in enumerate(v_arr):
                # print(i,v_t)
                v_t = str(v_t)
                if 'q=' in v_t and 'dim=' in v_t:
                    v_arr[i] = v_t.split('(')[0]
                elif 'q_samp' in v_t and 'std' in v_t:
                    ts = v_t.split('(')
                    min_t = min([float(t) for t in ts[1].split('|')[4].split('=')[1].split('+')])
                    v_arr[i] = ts[0] + '|' + f'{min_t:.4f}'
                else:
                    pass
        return v_arr

    with ExcelWriter(output_file) as writer:
        for i, sheet_name in enumerate(xls.sheet_names):
            print(i, sheet_name)
            df = pd.read_excel(input_file, sheet_name=sheet_name, header=None,
                               index_col=None)  # index_col=False: not use the first columns as index
            values = []
            for j, v in enumerate(df.values):
                values.append(process_line(v))
            # Generate dataframe from list and write to xlsx.
            pd.DataFrame(values).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            # styled_df = df_1.style.apply(highlight_cell, axis=1)
            # styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        writer.save()

    return output_file


def txt_to_csv(txt_file, csv_file):
    with open(csv_file, 'w') as out_f:
        head = "ts,uid,id.orig_h(srcIP),id.orig_p(sport),id.resp_h(dstIP),id.resp_p(dport),proto,service,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,tunnel_parents,label,detailed-label"
        out_f.write(head + '\n')
        with open(txt_file, 'r') as in_f:
            line = in_f.readline()
            while line:
                arr = line.split()
                l = ",".join(arr)
                out_f.write(l + '\n')

                line = in_f.readline()


def plot_bar_difference_seaborn_vs_stats(data=[], show_detectors=['OCSVM', 'AE'],
                                         show_datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'], gs=True,
                                         show_repres=['IAT', 'IAT-FFT', 'SIZE'],
                                         colors=['tab:brown', 'tab:green', 'm', 'c', 'b', 'r'],
                                         output_file="F1_for_all.pdf", xlim=[-0.1, 1]):
    # import seaborn as sns
    # sns.set_style("darkgrid")

    # create plots
    fig_cols = len(show_datasets) // 2 + 1
    figs, axes = plt.subplots(2, fig_cols)
    bar_width = 0.13
    opacity = 1

    s = min(len(show_repres), len(colors))
    for ind, dataset_name in enumerate(show_datasets):
        s_data = []
        for j, detector_name in enumerate(show_detectors):
            sub_dataset = []
            new_colors = []
            for i, (repres, color) in enumerate(zip(show_repres[:s], colors[:s])):
                if 'SAMP-' in repres.upper():
                    max_auc, min_auc = data[dataset_name][detector_name][gs][-1][i]
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                else:
                    aucs = [float(data[dataset_name][detector_name][gs][-1][i]), 0]
                if i == 0:
                    pre_aucs = aucs
                    continue
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                sub_dataset.append((detector_name, repres, diff[0]))
                # #rects = plt.bar((ind) + (i) * bar_width, height=diff[0], width=bar_width, alpha=opacity, color=color,
                #  #               label='Frank' + str(i))
                # # g = sns.barplot(x='day', y='tip', data=groupedvalues, palette=np.array(pal[::-1])[rank])
                # sns.boxplot(y="b", x="a", data=diff, orient='v', ax=axes[ind])
                # # autolabel(rects, aucs=aucs, pre_aucs=pre_aucs)
                # pre_aucs = aucs
                new_colors.append(color)
            s_data.extend(sub_dataset)

        if ind % fig_cols == 0:
            if ind == 0:
                t = 0
            else:
                t += 1
        df = pd.DataFrame(s_data, columns=['detector', 'repres', 'diff'])
        g = sns.barplot(y="diff", x='detector', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        g.set_title(dataset_name)
        g.get_legend().set_visible(False)

    # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=13)) for v in labels]
    pos1 = axes[-1, -1].get_position()  # get the original position
    # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # ax.set_position(pos2) # set a new position
    loc = (pos1.x0 + 0.05, pos1.y0 + 0.05)
    print(f'loc: {loc}, pos1: {pos1.bounds}')
    figs.legend(handles, labels, title='Representation', loc=loc,
                ncol=1, prop={'size': 8})  # loc='lower right',  loc = (0.74, 0.13)

    ind += 1
    while t < 2:
        if ind % fig_cols == 0:
            t += 1
            if t >= 2:
                break
            ind = 0
        # remove subplot
        # fig.delaxes(axes[1][2])
        axes[t, ind % fig_cols].set_axis_off()
        ind += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    # plt.tight_layout()
    #

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(output_file)  # should use before plt.show()
    plt.show()

    del fig  # must have this.

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})


def plot_bar_difference_seaborn(data=[], show_detectors=['OCSVM', 'AE'],
                                show_datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'], gs=True,
                                show_repres=['IAT', 'IAT-FFT', 'SIZE'],
                                colors=['tab:brown', 'tab:green', 'm', 'c', 'b', 'r'],
                                output_file="F1_for_all.pdf", xlim=[-0.1, 1], tab_type='', appendix=False):
    sns.set_style("darkgrid")
    print(show_detectors)
    # create plots
    num_figs = len(show_detectors)
    if not appendix:
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
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0
    s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    for j, detector_name in enumerate(show_detectors):
        sub_dataset = []
        new_colors = []
        yerrs = []
        for ind, dataset_name in enumerate(show_datasets):
            n = 100
            test_size = sum([int(v) for v in data[detector_name][dataset_name]['test_size']])
            yerr = []
            if dataset_name not in new_data.keys():
                new_data[dataset_name] = {detector_name: {}, 'test_size': [], 'yerr': []}

            if tab_type.upper() == "basic_representation".upper():
                features = ['STATS', 'IAT', 'IAT-FFT', 'SAMP-NUM', 'SAMP-NUM-FFT', 'SAMP-SIZE', 'SAMP-SIZE-FFT']
                f_dict = dict(zip(features, [i for i in range(len(features))]))
                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'IAT'
                B = 'IAT-FFT'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}
                print(dataset_name, detector_name, 'IAT:', pre_aucs, 'IAT-FFT:', aucs)

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-NUM'
                B = 'SAMP-NUM-FFT'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]
                # new_colors = ['b', 'r']
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-SIZE'
                B = 'SAMP-SIZE-FFT'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]
                # new_colors = ['b', 'r']
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 3

                # yerr = [1 / np.sqrt(test_size)] * 2

            elif tab_type.upper() == "effect_size".upper():
                features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP-SIZE']
                f_dict = dict(zip(features, [i for i in range(len(features))]))

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'IAT'
                B = 'IAT+SIZE'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-NUM'
                B = 'SAMP-SIZE'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 2
            elif tab_type.upper() == "effect_header".upper():
                features = ['STATS (wo. header)', 'STATS (w. header)', 'IAT+SIZE (wo. header)', 'IAT+SIZE (w. header)',
                            'SAMP-SIZE (wo. header)', 'SAMP-SIZE (w. header)']
                f_dict = dict(zip(features, [i for i in range(len(features))]))

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'STATS (wo. header)'
                B = 'STATS (w. header)'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()

                A = 'IAT+SIZE (wo. header)'
                B = 'IAT+SIZE (w. header)'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                # new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-SIZE (wo. header)'
                B = 'SAMP-SIZE (w. header)'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

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
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['repres'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=new_yerrs, fmt='none', c='b', capsize=3)
        # for p in g.patches:
        #     height = p.get_height()
        #     # print(height)
        #     if  height < 0:
        #         height= height - 0.03
        #         xytext = (0, 0)
        #     else:
        #         height = p.get_height() + 0.03
        #         xytext = (0, 0)
        #     g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
        #                    va='center', xytext=xytext, textcoords='offset points')

        # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        font_size = 20
        g.set_ylabel('AUC difference', fontsize=font_size + 4)
        if appendix:
            if j < len(show_detectors) - 1:
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
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(False)

    # # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # pos1 = axes[-1, -1].get_position()  # get the original position
    # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # ax.set_position(pos2) # set a new position
    # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)

    fig.legend(handles, labels, loc='lower center',  # upper left
               ncol=3, prop={'size': font_size - 2})  # l

    # figs.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper right', ncol=1)
    # remove subplot
    # fig.delaxes(axes[1][2])
    # axes[-1, -1].set_axis_off()

    # j += 1
    # while t < r:
    #     if j % c == 0:
    #         t += 1
    #         if t >= r:
    #             break
    #         j = 0
    #     # remove subplot
    #     # fig.delaxes(axes[1][2])
    #     axes[t, j % c].set_axis_off()
    #     j += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    plt.tight_layout()
    #

    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.35)
        else:
            if appendix:
                if "GMM" in ",".join(show_detectors):
                    plt.subplots_adjust(bottom=0.2)
                else:
                    plt.subplots_adjust(bottom=0.10)
            else:
                plt.subplots_adjust(bottom=0.18)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(output_file)  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return new_data, new_colors


def plot_bar_ML_seaborn(new_data=[], new_colors=[], show_detectors=[], gs=True,
                        output_file="F1_for_all.pdf", xlim=[-0.1, 1], tab_type=''):
    # import seaborn as sns
    sns.set_style("darkgrid")
    print(show_detectors)
    # create plots
    num_figs = len(new_data.keys())
    c = 3
    if num_figs >= c:
        if num_figs % c == 0:
            r = int(num_figs // c)
        else:
            r = int(num_figs // c) + 1  # in each row, it show 4 subplot
        figs, axes = plt.subplots(r, c, figsize=(65, 40))  # (width, height)
    else:
        figs, axes = plt.subplots(1, num_figs, figsize=(20, 8))  # (width, height)
        axes = axes.reshape(1, -1)
    print(f'subplots: ({r}, {c})')
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0
    for i, dataset_name in enumerate(new_data.keys()):
        sub_dataset = []
        yerrs = []
        for j, (detector_name, dv_dict) in enumerate(new_data[dataset_name].items()):
            if detector_name not in show_detectors:
                continue
            for _, (repres_pair, diff) in enumerate(dv_dict[gs].items()):
                sub_dataset.append((detector_name, repres_pair, float(diff)))
            yerrs.append(new_data[dataset_name]['yerr'])
        print(f'{dataset_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['detector', 'representation', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        if i % c == 0 and i > 0:
            t += 1
        g = sns.barplot(y="diff", x='detector', data=df, hue='representation', ax=axes[t, i % c],
                        palette=new_colors)  # palette=show_clo
        # for index, row in df.iterrows():
        #     g.text(row.name, row.representation, round(row.auc, 2), color='black', ha="center")
        yerrs = np.asarray(yerrs)
        new_yerrs = []
        for c_tmp in range(yerrs.shape[1]):  # extend by columns
            new_yerrs.extend(yerrs[:, c_tmp])
        print('new_yerrs for ML diff:', new_yerrs)

        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['representation'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['detector'].nunique() and i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.6)
                pre = cur + p.get_width() * num_bars

        axes[t, i % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=new_yerrs, fmt='none', c='b', capsize=3)

        # g.set(xlabel=dataset_name)
        # g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        # # g.set_title(detector_name, y=-0.005)
        # g.get_legend().set_visible(False)

        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        font_size = 95
        g.set_xticklabels(g.get_xticklabels(), fontsize=font_size - 5)
        # g.set_yticklabels(['{:,.2f}'.format(x) for x in g.get_yticks()], fontsize=font_size - 3)
        # print(g.get_yticks())
        g.set_ylabel('AUC difference', fontsize=font_size + 4)
        y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x % 0.5 == 0]
        g.set_yticks(y_v)  # set value locations in y axis
        g.set_yticklabels(y_v, fontsize=font_size - 3)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        if i % c != 0:
            g.get_yaxis().set_visible(False)

        g.set_title(dataset_name, fontsize=font_size)
        g.get_legend().set_visible(False)

    # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=26)) for v in labels]
    print('---', labels)
    pos1 = axes[-1, -1].get_position()  # get the original position
    # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # ax.set_position(pos2) # set a new position
    loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    print(f'loc: {loc}, pos1: {pos1.bounds}')

    # cnt_tmp = r * c - num_figs
    # if cnt_tmp == 0:
    #     last_num = 1
    #     labels = ["\n".join(textwrap.wrap(v, width=26)) for v in labels]
    # else:  # elif cnt_tmp > 0:
    #     last_num = 2
    # gs = axes[r - 1, -last_num].get_gridspec()  # (1, 1): the start point of the new merged subplot
    # for ax in axes[r - 1, -last_num:]:
    #     ax.remove()
    # axbig = figs.add_subplot(gs[r - 1, -last_num:])
    #
    # axbig.legend(handles, labels, loc=2,  # upper left
    #              ncol=1, prop={'size': font_size - 25})  # loc='lower right',  loc = (0.74, 0.13)
    # # figs.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper rig ht', ncol=1)
    figs.legend(handles, labels, loc='lower center',  # upper left
                ncol=3, prop={'size': font_size - 25})  # l

    # # remove subplot
    # # fig.delaxes(axes[1][2])
    # axbig.set_axis_off()

    i += 1
    while t < r:
        if i % c == 0:
            t += 1
            if t >= r:
                break
            i = 0
        # remove subplot
        # fig.delaxes(axes[1][2])
        axes[t, i % c].set_axis_off()
        i += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    plt.tight_layout()
    #
    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.subplots_adjust(bottom=0.1)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(output_file)  # should use before plt.show()
    plt.show()
    plt.close(figs)

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})


@func_notation
def save_total_results(results, output_file='xxx.txt', detector_name='GMM', params={}):
    """2.1 save the results to txt"""
    print(f'output_file: {output_file}')
    with open(output_file, 'w') as out_hdl:
        if 'file_header' in params.keys():
            out_hdl.write(params['file_header'] + '\n')
        save_dict(results, keys=[], out_hdl=out_hdl)

    """2.2 save the results_dict to dat
    """
    output_dat = output_file + '.dat'
    print(f'output_dat: {output_dat}')
    with open(output_dat, 'wb') as out_hdl:
        pickle.dump(results, out_hdl)

    """2.3 save all results to csv
    """
    output_csv = write2csv(results, detector_name=detector_name, output_file=output_dat + '.csv')
    print(f'output_csv: {output_csv}')

    """2.4 change the csv to xlsx
    """
    output_xlsx = csv2xlsx(output_csv, detector_name=detector_name, output_file=output_csv + '.xlsx')
    print(f'output_xlsx: {output_xlsx}')

    """2.41 highlight cell in the xlsx
        """
    output_highlight_xlsx = xlsx_highlight_cells(output_xlsx, output_file=output_xlsx + '_highlight.xlsx')
    print(f'output_highlight_xlsx: {output_highlight_xlsx}')

    """2.5 change csv to latex tables
    """
    output_tab = csv2latex_tab(detector_name, input_file=output_csv,
                               output_file=output_csv + '_tabs.txt', verbose=0)
    print(f'output_tab: {output_tab}')


def main():
    case = 'find_difference'
    case = 'csv2figs'
    out_dir = 'out/report/reprst_srcip/20201227'
    if case == 'write2csv':
        # # 1. xxx.dat to csv
        # output_file ='output_data/GMM_result_2020-01-28 20:22:05-tmp.txt.dat'
        detector_name, output_file = ('GMM', f'{out_dir}/GMM_result_2020-12-23 04:47:46.txt.dat')
        if os.path.exists(output_file):
            print('++++', output_file, flush=True)
        with open(output_file, 'rb') as out_hdl:
            results = pickle.load(out_hdl)
        print(results)
        file_name = write2csv(results, detector_name=detector_name)
        print(file_name)
    elif case == 'csv2xlsx':
        # 2. csv to xlsx
        # filename=write2xls(results)
        # # refresh(filename=filename)
        detector_name = 'KDE'
        input_file = f'../output_data/{detector_name}_results.txt'
        # input_file='../output_data/result_2020-01-04 12:41:53.txt'
        #
        results = load_result_data(input_file)
        filename = write2csv(results, detector_name=detector_name, output_file=input_file + '.csv')
        print(filename)
        # file_name = '../output_data/AE_result_2020-01-17 23:11:00.txt.dat.csv'
        # # file_name='../output_data/GMM_result_2020-01-28 20:22:05-tmp.txt.dat.csv'
        # file_name = csv2xlsx(file_name, output_file=file_name + '.xlsx')
        # print(file_name)
    elif case == 'highlight cell':
        # 3. highlight cell
        file_name = '../output_data/PCA_result_2020-03-04 00/53/53-tmp.txt.dat.csv.xlsx_highlight.xlsx'
        file_name = hightlight_excel(file_name, output_file=file_name + '.-part.xlsx')  # one-cell with different color
        file_name = xlsx_highlight_cells(file_name, output_file=file_name + '.all.xlsx')  # one-cell with one color
        print(file_name)
    elif case == 'find_difference':  # file_name_1 - file_name_2
        # # 4. find differences between two xlsx.
        # # # file_name = 'output_data/KDE-IF-Results-q_flow=0.90-20200130.xlsx'
        # dir_root = 'out'
        # file_name_1 = f'{dir_root}/output_data/AE_20200406.xlsx'
        input_files = [
            ("OCSVM", f'{out_dir}/OCSVM.txt.dat.csv.xlsx_highlight.xlsx'),
            ("KDE", f'{out_dir}/KDE.txt.dat.csv.xlsx_highlight.xlsx'),
            ("GMM", f'{out_dir}/GMM.txt.dat.csv.xlsx_highlight.xlsx'),
            ("IF", f'{out_dir}/IF.txt.dat.csv.xlsx_highlight.xlsx'),
            ("PCA", f'{out_dir}/PCA.txt.dat.csv.xlsx_highlight.xlsx'),
            ("AE", f'{out_dir}/AE.txt.dat.csv.xlsx_highlight.xlsx')
        ]
        output_file = f'{out_dir}/Results-merged.xlsx'
        file_name_1 = merge_xlsx(input_files, output_file)
        file_name_2 = f'{out_dir}/Results-highlight-20200509-srcip.xlsx'  #
        output_file = file_name_1 + '.diff.xlsx'

        # file_name_1 = f'out/report/reprst_srcip_new/20201227/Results-merged.xlsx'   # only src
        # file_name_2 = f'out/report/reprst/20201227/Results-merged.xlsx'  # src+dst data
        # # # file_name = xlsx_highlight_cells(file_name, output_file=file_name + '.highlight.xlsx')
        # output_file =  f'out/report/reprst_srcip_new/20201227/srcip-src_dst-diff.xlsx'
        file_name = diff_files(file_name_1, file_name_2, output_file)
        print(file_name)
        # # for i, detector_name in enumerate(['KDE', 'IF']):
        #     file_name = xlsx_highlight_cells(file_name, sheet_name=detector_name)
        #     print(file_name)
    elif case == 'csv2latex':
        # 5. csv to latex tables
        file_name = 'output_data/IF_result_2020-02-02 11:13:07-final.txt.dat.csv'
        """2.5 change csv to latex tables
           """
        output_tab = csv2latex_tab('IF', input_file=file_name,
                                   output_file=file_name + '_tabs.txt', verbose=0)
        print(f'output_tab: {output_tab}')
    elif case == 'keep AUC':
        # 6. only keep AUC in xlsx
        file_name = '../output_data/CTU-normal-malicious.xlsx'
        file_name = clean_xlsx(file_name, output_file=file_name + '_clean.xlsx')
    elif case == 'csv2latex2':
        output_file = out_dir + 'results_latex.txt'
        header = False
        num_feat = 3
        # if num_feat == 3:
        #     tab_latex = tab_latex_3
        # else:
        #     tab_latex = tab_latex_7
        # with open(output_file, 'w') as f:
        #     for i, (detector, input_file) in enumerate(input_files.items()):
        #         tab_false, tab_true = csv2latex_previous(input_file=input_file,
        #                                                  tab_latex=tab_latex, caption=detector, do_header=header,
        #                                                  previous_result=True,
        #                                                  num_feat=num_feat,
        #                                                  verbose=0)
        #
        #         # tab_false, tab_true = csv2latex(input_file=input_file,
        #         #                                 tab_latex=tab_latex, caption=detector, do_header=header,
        #         #                                 num_feat=num_feat,
        #         #                                 verbose=0)
        #
        #         f.write(detector + ':' + input_file + '\n')
        #         for line in tab_false:
        #             f.write(line + '\n')
        #         print()
        #
        #         for line in tab_true:
        #             f.write(line + '\n')
        #         print()
        #         f.write('\n')
    elif case == 'csv2figs':
        # file_name = '../output_data/AE_result_2020-03-05 00:11:02-final.txt.dat.csv.xlsx_highlight.xlsx'
        input_files = [
            ("OCSVM", f'{out_dir}/OCSVM.txt.dat.csv.xlsx_highlight.xlsx'),
            ("KDE", f'{out_dir}/KDE.txt.dat.csv.xlsx_highlight.xlsx'),
            ("GMM", f'{out_dir}/GMM.txt.dat.csv.xlsx_highlight.xlsx'),
            ("IF", f'{out_dir}/IF.txt.dat.csv.xlsx_highlight.xlsx'),
            ("PCA", f'{out_dir}/PCA.txt.dat.csv.xlsx_highlight.xlsx'),
            ("AE", f'{out_dir}/AE.txt.dat.csv.xlsx_highlight.xlsx')
        ]
        output_file = f'{out_dir}/Results-merged.xlsx'
        file_name = merge_xlsx(input_files, output_file)
        # file_name = f'{out_dir}/Results-highlight-20201225.xlsx'
        # file_name = hightlight_excel(file_name, output_file=file_name + '.-part.xlsx')  # one-cell with different color
        # file_name = xlsx_highlight_cells(file_name, output_file=file_name + '.all.xlsx')  # one-cell with one color

        appendex_flg = True
        if not appendex_flg:
            output_file = f'{out_dir}/res-best/All_latex_tables_figs.txt'  # for main paper results
            check_n_generate_path(output_file)
            with open(output_file, 'w') as f:
                for fig_flg in ['default',
                                'rest']:  #:  # 'default' (main paper results), rest(part of appendix results)
                    for gs in [True, False]:  # :  # gs=True ((best parameters), gs=False (default parameters)
                        for i, tab_type in enumerate(['basic_representation', 'effect_size',
                                                      'effect_header']):  # 'appd_all_without_header', 'appd_all_with_header', 'appd_samp', 'feature_dimensions']
                            # for i, tab_type in enumerate(['effect_header']):
                            print('\n\n******************')
                            print(i, tab_type, gs, fig_flg)
                            try:
                                # xlsx2latex_tab(file_name, output_file=file_name + "-" + tab_type + '-table_latex.txt',
                                #                tab_type=tab_type)
                                # fig_flg = 'default'
                                xlsx2latex_figure(f=f, input_file=file_name,
                                                  output_file=os.path.dirname(
                                                      output_file) + "/" + tab_type + '-' + fig_flg + '-figure_latex.txt',
                                                  tab_type=tab_type, fig_flg=fig_flg, gs=gs, appendix=appendex_flg)
                                # fig_flg = 'rest'
                                # xlsx2latex_figure(file_name, output_file=file_name + "-" + tab_type + '-' + fig_flg + '-figure_latex.txt',
                                #                   tab_type=tab_type, fig_flg=fig_flg, gs=gs)


                            except Exception as e:
                                print('Error: ', i, e)
                                traceback.print_exc()
                                continue

                    # release the matplotlib memory
                    # Clear the current figure.
                    plt.clf()
                    # Closes all the figure windows.
                    plt.close('all')

                    import gc
                    gc.collect()

        else:
            output_file = f'{out_dir}/res-appendix/All_latex_tables_figs-appendix.txt'  # for appendix results
            check_n_generate_path(output_file)
            with open(output_file, 'w') as f:
                for gs in [True, False]:  #
                    for i, tab_type in enumerate(['basic_representation', 'effect_size',
                                                  'effect_header']):  # 'appd_all_without_header', 'appd_all_with_header', 'appd_samp', 'feature_dimensions']
                        # for i, tab_type in enumerate(['effect_header']):
                        print('\n\n******************')
                        try:
                            # xlsx2latex_tab(file_name, output_file=file_name + "-" + tab_type + '-table_latex.txt',
                            #                tab_type=tab_type)
                            fig_flg = 'default'
                            xlsx2latex_figure(f=f, input_file=file_name,
                                              output_file=os.path.dirname(
                                                  output_file) + "/" + tab_type + '-' + fig_flg + '-figure_latex-appendix.txt',
                                              tab_type=tab_type, fig_flg=fig_flg, gs=gs, appendix=appendex_flg)


                        except Exception as e:
                            print('Error: ', i, e)
                            traceback.print_exc()
                            continue

                # release the matplotlib memory
                # Clear the current figure.
                plt.clf()
                # Closes all the figure windows.
                plt.close('all')

                import gc
                gc.collect()


if __name__ == '__main__':
    main()
