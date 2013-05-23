from shogun.Structure	import Sequence

import numpy

def get_label_arr(labels, i):
	'''
	TODO DOC
	'''
	# get StructuredData
	label = labels.get_label(i)
	# get Sequence
	seq = Sequence.obtain_from_generic(label)
	# get numpy.array
	arr = seq.get_data()

	return arr

def get_statistics(true_labels, predicted_labels):
	'''
	TODO DOC
	'''
	d = dict()

	# total number of elements in the sequences correctly predicted
	d['success_count'] = 0

	# state statistics
	d['true_0_count'] = 0
	d['true_1_count'] = 0
	d['pred_0_count'] = 0
	d['pred_1_count'] = 0

	for i in xrange(true_labels.get_num_labels()):
		# get the ground truth numpy.array of the ith example
		true_arr = get_label_arr(true_labels, i)
		# get the predicted numpy.array of the ith example
		pred_arr = get_label_arr(predicted_labels, i)

		d['true_0_count'] += numpy.sum(true_arr == 0)
		d['true_1_count'] += numpy.sum(true_arr == 1)

		d['pred_0_count'] += numpy.sum(pred_arr == 0)
		d['pred_1_count'] += numpy.sum(pred_arr == 1)

		d['success_count'] += numpy.sum(true_arr == pred_arr)

	return d
