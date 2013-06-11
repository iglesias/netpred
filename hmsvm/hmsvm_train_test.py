#!/usr/bin/env python

import	utils

from shogun.Evaluation	import StructuredAccuracy
from shogun.Loss		import HingeLoss
from shogun.Structure	import HMSVMModel, SMT_TWO_STATE, PrimalMosekSOSVM, DualLibQPBMSOSVM
from modshogun			import MSG_DEBUG

### prepare training, test data and evaluator

'''
train_data_file = 'hmm_gaussian_0_lb_1_ub_fold'
train_num_examples_fold = 20
train_num_folds = 5
train_labels, train_features = utils.unfold_data(train_data_file)
'''

train_data_file = 'hmsvm_30_distort_5_examples_data'
train_num_examples = 5
train_labels, train_features = utils.read_mat_file(train_data_file, train_num_examples)

test_data_file = 'hmsvm_30_distort_data_test'
test_num_examples = 100
test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

evaluator = StructuredAccuracy()

### train HM-SVM and evaluate in training data

print 'training HM-SVM'

model = HMSVMModel(train_features, train_labels, SMT_TWO_STATE)
model.set_use_plifs(True)

loss = HingeLoss()

# regularizer chosen performing cross-validation
regularizer = 5.0

sosvm = PrimalMosekSOSVM(model, loss, train_labels)
sosvm.set_regularizer(5)
sosvm.train()
prediction = sosvm.apply()
accuracy = evaluator.evaluate(prediction, train_labels)
print '\ttraining accuracy: ' + str(accuracy*100) + '%'
utils.print_statistics(train_labels, prediction)

### evaluate in test data

print 'testing HM-SVM'
prediction = sosvm.apply(test_features)
accuracy = evaluator.evaluate(prediction, test_labels)
print '\ttest accuracy: ' + str(accuracy*100) + '%'
utils.print_statistics(test_labels, prediction)
