#!/usr/bin/env python

import	utils
import numpy

from mlhmm		import MLHMM
from itertools	import product

from shogun.Evaluation	import StructuredAccuracy
from shogun.Structure	import HMSVMModel, SMT_TWO_STATE

### prepare training, test data and evaluator

train_data_file = 'hmsvm_30_distort_5_examples_data'
train_num_examples = 5
train_labels, train_features = utils.read_mat_file(train_data_file, train_num_examples)

test_data_file = 'hmsvm_30_distort_data_test'
test_num_examples = 100
test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

evaluator = StructuredAccuracy()

### train ML-HMM and evaluate in training data

print 'training ML-HMM'
model = HMSVMModel(train_features, train_labels, SMT_TWO_STATE)
model.set_use_plifs(True)
mlhmm = MLHMM(model)
mlhmm.train()

'''
print '\tmodel parameters:'
print '\t- transition scores: ' + str(numpy.exp(mlhmm.transition_scores))
print '\t- feature scores:'
for s,f in product(xrange(mlhmm.num_free_states), xrange(mlhmm.num_features)):
	print '\t\tstate %d feature %d:\n%s' % (s, f, str(numpy.exp(mlhmm.feature_scores[f,s,:])))
'''

prediction = mlhmm.apply()
accuracy = evaluator.evaluate(prediction, train_labels)
print '\ttraining accuracy: ' + str(accuracy*100) + '%'
utils.print_statistics(train_labels, prediction)

### evaluate in test data

print 'testing ML-HMM'
prediction = mlhmm.apply(test_features)
accuracy = evaluator.evaluate(prediction, test_labels)
print '\ttest accuracy: ' + str(accuracy*100) + '%'
utils.print_statistics(test_labels, prediction)
