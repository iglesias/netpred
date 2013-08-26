#!/usr/bin/env python

from shogun.Structure	import TwoStateModel, HMSVMModel, SMT_TWO_STATE, DualLibQPBMSOSVM, Sequence
from shogun.Structure	import SequenceLabels
from shogun.Loss		import HingeLoss
from shogun.Features	import RealMatrixFeatures
from shogun.Evaluation	import StructuredAccuracy
from modshogun			import MSG_DEBUG

import numpy
import scipy.io
import pickle
import utils
import time

checks = False

# number of folds used for model selection
K = 5
# number of examples per fold
num_fold_examples = 20
# length of each example
example_len = 250
# number of features per example
num_features = 10
# number of features that are pure noise
num_noise_features = 3
# the number different label values
num_states = 2
# K models that will contain the data of each fold
models = []

data_file = 'hmm_gaussian_3_lb_4_ub_fold'
for k in xrange(K):
	data_dict = scipy.io.loadmat('/home/nando/workspace/hmsvmToolbox-0.2/src/hmsvm_toydata/%s_%d.mat' % (data_file, k), struct_as_record=False)

	labels_array = data_dict['label'][0]
	idxs = numpy.nonzero(labels_array == -1)
	labels_array[idxs] = 0

	labels = SequenceLabels(labels_array.astype(numpy.int32), example_len, num_fold_examples, num_states)
	features = RealMatrixFeatures(data_dict['signal'].astype(float), example_len, num_fold_examples)
	models.append(HMSVMModel(features, labels, SMT_TWO_STATE))

# check
if checks:
	print 'running checks on simulated data'
	for k in xrange(K):
		labels = models[k].get_labels()
		features = RealMatrixFeatures.obtain_from_generic(models[k].get_features())

		print '\tmodel %d with %d labels and %d features' % (k, labels.get_num_labels(),
			features.get_num_vectors())
		assert(labels.get_num_labels() == features.get_num_vectors())

		for i in xrange(labels.get_num_labels()):
			label = Sequence.obtain_from_generic(labels.get_label(i))
			feature_vector = features.get_feature_vector(i)
			assert(label.get_data().shape[0] == feature_vector.shape[1])

	print 'the simulated data is OK!\n'

labels_no_fold   = []
features_no_fold = []
# for each fold
for k1 in xrange(K):

	# put all labels/features together except the ones of the current fold
	labels_no_kfold = SequenceLabels(num_fold_examples*(K-1), num_states)
	features_no_kfold = RealMatrixFeatures(num_fold_examples*(K-1), num_features)
	# index for the next feature vector to set
	idx = 0

	for k2 in xrange(K):
		if k1 != k2:
			labels_k_fold = models[k2].get_labels()
			features_k_fold = RealMatrixFeatures.obtain_from_generic(models[k2].get_features())
			assert(labels_k_fold.get_num_labels() == features_k_fold.get_num_vectors())

			for i in xrange(labels_k_fold.get_num_labels()):
				labels_no_kfold.add_label(labels_k_fold.get_label(i))
				features_no_kfold.set_feature_vector(features_k_fold.get_feature_vector(i), idx)
				idx += 1

	labels_no_fold.append(labels_no_kfold)
	features_no_fold.append(features_no_kfold)

# check
if checks:
	print 'running checks on folded data'
	for k in xrange(K):
		print '\tin labels_no_fold   %d there are %d labels' % (k, labels_no_fold[k].get_num_labels())
		print '\tin features_no_fold %d there are %d feature vectors' % (k,
				features_no_fold[k].get_num_vectors())

		assert(labels_no_fold[k].get_num_labels() == features_no_fold[k].get_num_vectors())

		# check that the length of labels and corresponding features are the same
		for i in xrange(labels_no_fold[k].get_num_labels()):
			label = Sequence.obtain_from_generic(labels_no_fold[k].get_label(i))
			feature_vector = features_no_fold[k].get_feature_vector(i)
			assert(label.get_data().shape[0] == feature_vector.shape[1])

	print 'the folds are OK!\n'

# structured SVM loss
loss = HingeLoss()

# accuracy evaluator
evaluator = StructuredAccuracy()

# regularization values
regularizers = [500, 50, 5, 0.5, 0.05]

W = []

for reg in regularizers:
	print 'training SO-SVM with regularization %.2f' % reg
	accuracies = []
	w = []
	for k in xrange(K):
		model = HMSVMModel(features_no_fold[k], labels_no_fold[k], SMT_TWO_STATE)
		model.set_use_plifs(True)
		sosvm = DualLibQPBMSOSVM(model, loss, labels_no_fold[k], reg*(K-1)*num_fold_examples)
		sosvm.set_verbose(True)
		print '\ton fold %d' % k
		t0 = time.time()
		sosvm.train()
		print '\t\tElapsed: training took ' + str(time.time()-t0)
		w.append(sosvm.get_w())
		t1 = time.time()
		prediction = sosvm.apply(models[k].get_features())
		print '\t\tElapsed: prediction took ' + str(time.time()-t1)
		accuracy = evaluator.evaluate(prediction, models[k].get_labels())
		print str(accuracy*100) + '%'
		statistics = utils.get_statistics(models[k].get_labels(), prediction)
		custom_accuracy = (100.*statistics['success_count'])/(num_fold_examples*example_len)
		print '\t\t%.2f\t1s: (%5d, %5d)\t0s: (%5d, %5d)' % (custom_accuracy,
				statistics['true_1_count'], statistics['pred_1_count'],
				statistics['true_0_count'], statistics['pred_0_count'])
		accuracies.append(accuracy)
	print '\toverall success rate of ' + str(numpy.mean(accuracies)*100) + '%'
	W.append(w)

pickle.dump(W, open('W_DualLibQPBMSOSVM_%s.p' % data_file, 'wb'))
