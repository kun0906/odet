""" Base class for report

"""


def parse_csv(in_file):
	""" parse csv file and store the data into dictionary

	Parameters
	----------
	in_file

	Returns
	-------
	results: dict
	"""
	def add_value(res, header, tuning, feature, dataset, model, value):
		if header not in res.keys():
			res[header] = {tuning: {feature: {dataset: {model: value}}}}
		if tuning not in res[header].keys():
			res[header][tuning] = {feature: {dataset: {model: value}}}
		if feature not in res[header][tuning].keys():
			res[header][tuning][feature] = {dataset: {model: value}}
		if dataset not in res[header][tuning][feature].keys():
			res[header][tuning][feature][dataset] = {model: value}
		if model not in res[header][tuning][feature][dataset].keys():
			res[header][tuning][feature][dataset][model] = value
		return res

	result = {}
	with open(in_file, 'r') as f:
		line = f.readline()
		while line:
			line = line.split(',')[:8]
			# [header, tuning, feature, dataset, model, f'{score:.2f}', shape, str(dim)]
			try:
				dataset, feature, header, model, tuning, score, shape, dim = line
				# shape = '|'.join([str(v.split('_')[0]) for v in data.split('|')])
				# dim = data.split('|')[-1].split('_')[-1]
				result = add_value(result, header, tuning, feature, dataset, model, (score, shape, dim))
			except Exception as e:
				print(f'Error: {e}, {line}')
			line = f.readline()

	return result


def format_name(data, data_orig2name={}):
	""" restore the data

	Parameters
	----------
	data
	data_orig2name

	Returns
	-------
	result: dict
	"""
	result = {}
	for header in data.keys():
		result[header] = {}
		for tuning in data[header].keys():
			result[header][tuning] = {}
			for feature in data[header][tuning].keys():
				result[header][tuning][feature] = {}
				for dataset in data[header][tuning][feature].keys():
					d = data_orig2name[dataset]
					result[header][tuning][feature][d] = {}
					for model, vs in data[header][tuning][feature][dataset].items():
						result[header][tuning][feature][d][model] = vs
	return result


