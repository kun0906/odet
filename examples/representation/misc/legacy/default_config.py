"""Includes all configurations, such as constants and global random_state.
    1. set a random seed for os, random, and so on.
    2. print control
    3. root directory control
    4. some constants
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

#############################################################################################
# 1. random state control in order to achieve reproductive results
# ref: https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
import datetime

START_TIME = datetime.datetime.now()
print(START_TIME)
# Seed value
# Apparently you may use different seed values at each stage
random_state = 42
# 1). Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(random_state)

# 2). Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(random_state)

# 3). Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(random_state)

# 4). set torch
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#############################################################################################
# 2. Print control

# import sys
#
# # force it to print everything in the buffer to the terminal immediately.
# sys.stdout.flush()
# # out_file = 'stdout_content.txt'
# # # sys.stdout = open(out_file, mode='w', buffering=1)
# # ###skip 'buffering' if you don't want the output_data to be flushed right away after written
# # # sys.stdout = sys.__stdout__
# #
# import functools
#
# print = functools.partial(print, flush=True)

"""Replace "print" with "logging"
# 
# The only time that print is a better option than logging is when the goal is to display a help statement
# for a command-line application. Other reasons why logging is better than print:
# 
# Ref:
#     https://docs.python.org/3/library/logging.html#logrecord-attributes
#     https://docs.python.org/3/library/logging.html#logrecord-attributes
# """
# import os
# import os.path as pth
# import sys
# import datetime
# import logging
# import logging.handlers as hdl
# from shutil import copy2 as cp  # use "copy2" instead of "copyfile"
# import colorlog
# from colorlog import ColoredFormatter, TTYColoredFormatter
#
# logger = logging.getLogger()
#
# # output log into stdout
# console_hdl = logging.StreamHandler(sys.stdout)
# # console_fmt = logging.Formatter('%(asctime)s %(name)-5s [%(threadName)-10s] %(levelname)-5s %(funcName)-5s '
# #                                 '%(filename)s line=%(lineno)-4d: %(message)s')
# # Add colorlog
# console_fmt = ColoredFormatter("%(log_color)s%(asctime)s %(name)-5s [%(threadName)-10s] %(levelname)-5s "
#                                "%(funcName)-5s %(filename)s line=%(lineno)-4d: %(message_log_color)s%(message)s",
#                                secondary_log_colors={'message': {'ERROR': 'red', 'CRITICAL': 'red'}})
# console_hdl.setFormatter(console_fmt)
# logger.addHandler(console_hdl)
#
# # output log into file
# dir_log = './log/data_reprst'
# pth_log = pth.join(dir_log, f'app_{START_TIME}.log')
# if not pth.exists(dir_log):
#     os.makedirs(dir_log)
# # if pth.exists(pth_log):
# #     cp(pth_log, pth_log+'time')
# # BackupCount: if either of maxBytes or backupCount is zero, rollover never occurs
# file_hdl = hdl.RotatingFileHandler(pth_log, mode='w', maxBytes=50 * 1024 * 1024,  # 500 * 1024 * 1024,
#                                    backupCount=20, encoding=None, delay=False)
# # file_hdl = hdl.TimedRotatingFileHandler(pth_log, when='H', interval=1, backupCount=10, encoding=None,
# #                                         delay=False, utc=False, atTime=None)
# # https://github.com/borntyping/python-colorlog/issues/64
# # colorlog doesn't strip colors when redirected to file for ColoredFormatter
# file_fmt = TTYColoredFormatter("%(log_color)s%(asctime)s %(name)-5s [%(threadName)-10s] %(levelname)-5s "
#                                "%(funcName)-5s %(filename)s line=%(lineno)-4d: %(message_log_color)s%(message)s",
#                                secondary_log_colors={'message': {'ERROR': 'red', 'CRITICAL': 'red'}},
#                                stream=file_hdl.stream)  # Both of them have the same format
# file_hdl.setFormatter(console_fmt)  # TTYColoredFormatter cannot work with RotatingFileHandler
# logger.addHandler(file_hdl)
#
#
# # https://streamhacker.com/2010/04/08/python-logging-filters/
# # https://stackoverflow.com/questions/18911737/using-python-logging-module-to-info-messages-to-one-file-and-err-to-another-file
# class LogFilter(logging.Filter):
#     def __init__(self, levelno=logging.INFO):
#         self.levelno = levelno
#
#     def filter(self, rec):
#         return rec.levelno == self.levelno
#
#
# # https://medium.com/@galea/python-logging-example-with-color-formatting-file-handlers-6ee21d363184
# # Output warning log
# fw = logging.FileHandler(pth_log + '-warning.log', mode='w')
# fw.setLevel(logging.WARNING)
# fw.setFormatter(file_fmt)
# fw.addFilter(LogFilter(levelno=logging.WARNING))  # only show warning log
# logger.addHandler(fw)
#
# # Output error log
# fe = logging.FileHandler(pth_log + '-error.log', mode='w')
# fe.setLevel(logging.ERROR)
# fe.setFormatter(file_fmt)  # has log whose level is larger than logging.ERROR
# logger.addHandler(fe)
#
# # set log level
# # log_key = 'INFO'
# # log_lv = {'DEBUG': (logging.DEBUG, logger.debug), 'INFO': (logging.INFO, logger.info)}
# # level, lg = log_lv[log_key]
# level = 'DEBUG'
# logger.setLevel(level)
# lg = logging.info
# ld = logging.debug
# lw = logging.warning
# le = logging.error
# lc = logging.critical

# ld('often makes a very good meal of %s', 'visiting tourists')
# logger.critical("OH NO everything is on fire")

import pickle

#
# """Step 3. Path control
# """
# import sys
# import warnings
#
# # add 'itod' root directory into sys.path
# root_dir = os.getcwd()
# sys.path.append(root_dir)
# # print(f'sys.path: {sys.path}')
# # check the workspace directory for 'open' function.
# # Note this directory can be different with the sys.path root directory
# workspace_dir = os.path.abspath(os.path.join(root_dir))
# if os.path.abspath(os.getcwd()) != workspace_dir:
#     msg = f'current directory does not equal workspace. Changing it to \'{workspace_dir}\'.'
#     warnings.warn(msg)
#     os.chdir(workspace_dir)
# lg(f'workspace_dir: {workspace_dir}')
#
#
# # 0. add work directory into sys.path
# import os.path as pth
# import sys
#
# # add root_path into sys.path in order you can access all the folders
# # avoid using os.getcwd() because it won't work after you change into different folders
# # NB: avoid using relative path
# root_path = pth.dirname(pth.dirname(pth.dirname(pth.abspath(__file__))))
# print(root_path)
# if f"{root_path}/examples" !=pth.abspath('.'):
#     raise ValueError(f"dir ({root_path}) is not correct, you should run the xxx.py under \'examples\'")
# sys.path.insert(0, root_path)
# # add root_path/examples into sys.path
# sys.path.insert(1, f"{root_path}/examples")
# # print(sys.path)   # for debug
#

#############################################################################################
"""4. Constant
"""
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# # set global variables
# # https://stackoverflow.com/questions/11813287/insert-variable-into-global-namespace-from-within-a-function
# # way1: global pth_log, ld, lg, lw, le, lc #(not work)
# # way2: globals()['lg']=lg  # (not work)
# import builtins
#
# builtins.ld = ld
# builtins.lg = lg
# builtins.lw = lw
# builtins.le = le
# builtins.lc = lc
