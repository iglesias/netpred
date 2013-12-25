#!/usr/bin/env python

import utils

from modshogun import *

distort = 40
evaluator = StructuredAccuracy()
regularizations = [500, 50, 5, 0.5, 0.005]
print '>>>> the distortion is ' + str(distort)

for regularization in regularizations:
#	print 'data with ' + str(distort) + '% of total labels distorted'
	print 'the regularization is ' + str(regularization) 

	### prepare training, test data and evaluator

	train_data_file = 'hmsvm_%d_distort_data_fold' % distort
	train_num_examples_fold = 20
	train_num_folds = 0.5
	train_labels, train_features = utils.unfold_data(train_data_file)

	test_data_file = 'hmsvm_%d_distort_data_test' % distort
	test_num_examples = 100
	test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

	### train HM-SVM and evaluate in training data

	model = HMSVMModel(train_features, train_labels, SMT_TWO_STATE)
	model.set_use_plifs(True)

	loss = HingeLoss()

	# regularization chosen performing cross-validation
	sosvm = PrimalMosekSOSVM(model, loss, train_labels)
	sosvm.set_regularization(regularization)
	sosvm.train()

	prediction = sosvm.apply()
	accuracy = evaluator.evaluate(prediction, train_labels)
	print '\ttraining accuracy:\t' + str(accuracy*100) + '%'
	utils.print_statistics(train_labels, prediction)

	### evaluate in test data

	prediction = sosvm.apply(test_features)
	accuracy = evaluator.evaluate(prediction, test_labels)
	print '\ttest accuracy:\t\t' + str(accuracy*100) + '%'
	utils.print_statistics(test_labels, prediction)
