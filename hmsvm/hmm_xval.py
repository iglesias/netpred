#!/usr/bin/env python

import numpy
import scipy
import utils

from mlhmm  import MLHMM
from scipy  import io
from time   import time

from shogun.Evaluation  import StructuredAccuracy
from shogun.Features    import RealMatrixFeatures
from shogun.Structure   import SequenceLabels, HMSVMModel, SMT_TWO_STATE
from modshogun          import MSG_DEBUG

### prepare data

# number of folds used for model selection
K = 5
# number of examples per fold
num_fold_examples = 20
# length of each example
example_len = 250
# number of features per example
num_features = 10
# the number different label values
num_states = 2
# K models that will contain the data of each fold
models = []

# load each data fold in a HMSVMModel
data_file = 'hmsvm_30_distort_data_fold'
for k in xrange(K):
	labels, features = utils.read_mat_file('%s_%d' % (data_file, k), num_fold_examples)
	models.append(HMSVMModel(features, labels, SMT_TWO_STATE))

# put together folds, leaving out one of them for each set
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

### training and validation

evaluator = StructuredAccuracy()
evaluator.io.set_loglevel(MSG_DEBUG)
accuracies = []

print 'training HMM'
for k in xrange(K):
	model = HMSVMModel(features_no_fold[k], labels_no_fold[k], SMT_TWO_STATE)
	model.set_use_plifs(True)
	hmm = MLHMM(model)
	print '\ton fold %d' % k,
	hmm.train()
	prediction = hmm.apply(models[k].get_features())
	accuracy = evaluator.evaluate(prediction, models[k].get_labels())
	print str(accuracy*100) + '%'
	utils.print_statistics(models[k].get_labels(), prediction)
	accuracies.append(accuracy)

print 'overall success rate of ' + str(numpy.mean(accuracies)*100) + '%'
