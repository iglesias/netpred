#!/usr/bin/env python

import utils

from mlhmm import MLHMM
from modshogun import StructuredAccuracy, HMSVMModel, SMT_TWO_STATE

distorts = [0, 10, 20, 30, 40]
evaluator = StructuredAccuracy()

for distort in distorts:
	print 'data with ' + str(distort) + '% of total labels distorted'

	### prepare training, test data and evaluator

	train_data_file = 'hmsvm_%d_distort_data_fold' % distort
	train_num_examples_fold = 20
	train_num_folds = 5
	train_labels, train_features = utils.unfold_data(train_data_file)

	test_data_file = 'hmsvm_%d_distort_data_test' % distort
	test_num_examples = 100
	test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

	### train ML-HMM and evaluate in training data

	model = HMSVMModel(train_features, train_labels, SMT_TWO_STATE)
	model.set_use_plifs(True)
	mlhmm = MLHMM(model)
	mlhmm.train()

	prediction = mlhmm.apply()
	accuracy = evaluator.evaluate(prediction, train_labels)
	print '\ttraining accuracy:\t' + str(accuracy*100) + '%'
	utils.print_statistics(train_labels, prediction)

	### evaluate in test data

	prediction = mlhmm.apply(test_features)
	accuracy = evaluator.evaluate(prediction, test_labels)
	print '\ttest accuracy:\t\t' + str(accuracy*100) + '%'
	utils.print_statistics(test_labels, prediction)
