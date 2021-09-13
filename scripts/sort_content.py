""" Main entry point:

	Instructions:
		# Note: run the below command under the root directory of the project.
		python3.7 scripts/sort_requirement.py

"""
import os
from shutil import copyfile


def format_txt(in_file):

	# 1. backup the in_file
	in_dir, file_name = os.path.split(in_file)
	orig_file = os.path.join(in_dir, '~trash', file_name)
	copyfile(in_file, orig_file)

	# 2. extract and sort data
	with open(in_file, 'r') as f:
		data = f.readlines()
		data = list(set(data))
		data.sort(key=lambda line: line.lower())  # sort the lines
		data = ''.join(data)
	print(data)

	# 3. save sorted data to in_file
	with open(in_file, 'w') as f:
		f.write(data)

	print(f'orig_file: {orig_file}\n'
	      f'sorted_file: {in_file}')


if __name__ == '__main__':
	format_txt(in_file='requirements.txt')
