#!/usr/bin/env python

import utils
import numpy

from mlhmm import MLHMM
from itertools import product
from modshogun import StructuredAccuracy, HMSVMModel, SMT_TWO_STATE

### prepare model, test data and evaluator

transition_scores, feature_scores, limits = utils.read_hmm_mat_file()

test_data_file = 'hmm_test_data'
test_num_examples = 100
test_labels, test_features = utils.read_mat_file(test_data_file, test_num_examples)

evaluator = StructuredAccuracy()

### set ground truth model

gthmm = MLHMM(transition_scores, feature_scores, limits)

### evaluate in test data

print 'testing GT-HMM'
prediction = gthmm.apply(test_features)
accuracy = evaluator.evaluate(prediction, test_labels)
print '\ttest accuracy: ' + str(accuracy*100) + '%'
utils.print_statistics(test_labels, prediction)
