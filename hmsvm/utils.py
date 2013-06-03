from shogun.Features	import RealMatrixFeatures
from shogun.Structure	import Sequence, SequenceLabels, HMSVMModel, SMT_TWO_STATE

import numpy
import scipy

from itertools	import product
from scipy		import io

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

def print_statistics(true_labels, predicted_labels):
	'''
	Calls get_statistics and prints them out.
	'''
	statistics = get_statistics(true_labels, predicted_labels)

	print '\t\t1s: (%5d, %5d)\t0s: (%5d, %5d)' % (
			statistics['true_1_count'], statistics['pred_1_count'],
			statistics['true_0_count'], statistics['pred_0_count'])

def read_mat_file(data_file, num_examples, example_len=250, num_states=2):
	'''
	Read labels and features from mat file, changes labels with value equal to -1 to 0 and returns
	SequenceLabels and RealMatrixFeatures objects.
	'''
	data_dict = scipy.io.loadmat('/home/nando/workspace/hmsvmToolbox-0.2/src/hmsvm_toydata/%s.mat' %
			(data_file), struct_as_record=False)

	labels_array = data_dict['label'][0]
	idxs = numpy.nonzero(labels_array == -1)
	labels_array[idxs] = 0

	labels = SequenceLabels(labels_array.astype(numpy.int32), example_len, num_examples, num_states)
	features = RealMatrixFeatures(data_dict['signal'].astype(float), example_len, num_examples)

	return labels, features

def read_hmm_mat_file(hmm_file='hmm', num_states=2, num_features=10, num_plif_nodes=20):
	'''
	Read transition, PLiF feature scores and supporting nodes from a HMM mat file.
	'''
	data_dict = scipy.io.loadmat('/home/nando/workspace/hmsvmToolbox-0.2/src/%s.mat' %
			(hmm_file), struct_as_record=False)

	hmm = data_dict['hmm']
	hmm = hmm[0,0]

	transition_scores = numpy.zeros(num_states**2)
	# transformation hardcoded -- WARNING: It will not work for HMMs with more than two states
	transition_scores[0] = hmm.trans_scores[0,0]
	transition_scores[1] = hmm.trans_scores[0,1]
	transition_scores[2] = hmm.trans_scores[1,1]
	transition_scores[3] = hmm.trans_scores[1,0]

	feature_scores = numpy.zeros((num_features, num_states, num_plif_nodes))
	for s,f in product(xrange(num_states), xrange(num_features)):
		feature_scores[f,s,:] = hmm.score_plifs[f,s].scores

	limits = hmm.score_plifs[0,0].limits

	return transition_scores, feature_scores, limits

def unfold_data(data_file, K=5, example_len=250, num_fold_examples=20, num_states=2, num_features=10):
	'''
	Put together K folds whose root name is data_file. In other words, it will read features and
	labels from the K files data_file_0.dat, data_file_1.dat, ..., data_file_K-1.dat, put them
	together in SequenceLabels and RealMatrixFeatures objects and return them.
	'''
	models = []
	### read mat files

	for k in xrange(K):
		labels, features = read_mat_file('%s_%d'%(data_file,k), num_fold_examples, example_len, num_states)
		models.append(HMSVMModel(features, labels, SMT_TWO_STATE))

	### put together folds

	labels_all = SequenceLabels(num_fold_examples*K, num_states)
	features_all = RealMatrixFeatures(num_fold_examples*K, num_features)
	# index for the next feature vector to set
	idx = 0
	for k in xrange(K):
		labels_k_fold = models[k].get_labels()
		features_k_fold = RealMatrixFeatures.obtain_from_generic(models[k].get_features())
		assert(labels_k_fold.get_num_labels() == features_k_fold.get_num_vectors())

		for i in xrange(labels_k_fold.get_num_labels()):
			labels_all.add_label(labels_k_fold.get_label(i))
			features_all.set_feature_vector(features_k_fold.get_feature_vector(i), idx)
			idx += 1

	return labels_all, features_all
