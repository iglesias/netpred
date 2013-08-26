#!/usr/bin/env python

import	utils

from shogun.Evaluation	import StructuredAccuracy
from shogun.Loss		import HingeLoss
from shogun.Structure	import HMSVMModel, SMT_TWO_STATE, DualLibQPBMSOSVM
from modshogun			import MSG_DEBUG

### prepare training, test data and evaluator

train_data_file = 'hmm_gaussian_3_lb_4_ub_fold'
train_num_examples_fold = 20
train_num_folds = 5
train_labels, train_features = utils.unfold_data(train_data_file)

test_data_file = 'hmm_gaussian_3_lb_4_ub_test'
test_num_examples = 100
test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

evaluator = StructuredAccuracy()

### train HM-SVM and evaluate in training data

print 'training HM-SVM'

model = HMSVMModel(train_features, train_labels, SMT_TWO_STATE)
model.set_use_plifs(True)

loss = HingeLoss()

# regularizer chosen performing cross-validation
regularizer = 0.5

sosvm = DualLibQPBMSOSVM(model, loss, train_labels, regularizer*train_num_folds*train_num_examples_fold)
# set debug traces
sosvm.set_verbose(MSG_DEBUG)
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
