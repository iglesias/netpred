#!/usr/bin/env python

import numpy as np
import util
import inference_methods
import matplotlib.pyplot	as pyplot

try:
	from shogun.Structure import DirectorStructuredModel
except ImportError:
	print "recompile shogun with --enable-swig-directors"
	import sys
	sys.exit(0)

from shogun.Features	import RealMatrixFeatures
from shogun.Library		import RealVector
from shogun.Structure	import SequenceLabels, ResultSet, DirectorStructuredModel, \
								Sequence, StructuredModel

class GridCRFStructuredModel(DirectorStructuredModel):
	'''
	Grid CRF with symmetric pairwise potentials. The most interesting parts are
	get_joint_feature_vector (the infamous \Psi!) and the argmax (also known
	as inference in PGM's argot).
	'''
	def __init__(self, features, labels, only_unaries):
		DirectorStructuredModel.__init__(self)
		self.only_unaries = only_unaries
		self.set_features(features)
		self.set_labels(labels)
		if self.only_unaries:
			self.dim = features.get_num_features()*labels.get_num_states()
		else:
			self.dim = features.get_num_features()*labels.get_num_states() + \
						labels.get_num_states()*(labels.get_num_states() + 1)/2
		self.n_states = labels.get_num_states()
		self.n_features = features.get_num_features()

	def init_training(self):
		return

	def get_dim(self):
		return self.dim

	def init_opt(self, regularization, A, a, B, b, lb, ub, C):
		# Magic value - it would be nicer to apply cross-validation
		C = 0.01*np.eye(self.dim)

	def check_training_setup(self):
		return True

	def get_num_aux(self):
		return 0

	def get_num_aux_con(self):
		return 0

	def check_training_setup(self):
		return True

	def get_joint_feature_vector(self, feat_idx, y):
		'''
		Compute the joint feature vector associated to the label y and the
		observation indeded by feat_idx.
		'''
		dummy.io.message(MSG_DEBUG, '', 0, 'Entering '
						'GridCRFStructuredModel::get_joint_feature_vector.\n')

		y_seq_arr = Sequence.obtain_from_generic(y).get_data()
		size = np.floor(np.sqrt(y_seq_arr.shape[0]))
		if (size**2,) != y_seq_arr.shape:
			raise ValueError('GridCRFStructuredModel limited to square grids, '
							 'the CRF has %d nodes instead.' % y_seq_arr.shape[0])
		n_nodes = np.int(size**2)

		features = RealMatrixFeatures.obtain_from_generic(self.get_features())
		feat_vec = features.get_feature_vector(feat_idx)
		# Transpose, because we want rol-major
		feat_vec = feat_vec.T
		self._check_feat_shape_(feat_vec, n_nodes)

		# Make square grid edges
		edges = util.make_grid_edges(size)

		# Arrange unary marginals
		gx = np.ogrid[:n_nodes]
		unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
		unary_marginals[gx, y_seq_arr] = 1

		# Compute accumulated unary marginals
		unary_acc = np.dot(unary_marginals.T, feat_vec)

		# Compute accumulated pairwise marginals forcing symmetry
		pairwise_acc = np.dot(unary_marginals[edges[:,0]].T, unary_marginals[edges[:,1]])
		pairwise_acc = pairwise_acc + pairwise_acc.T - np.diag(np.diag(pairwise_acc))

		# Stack together unaries and pairwise
		if self.only_unaries:
			psi = np.hstack(unary_acc.ravel())
		else:
			psi = np.hstack([unary_acc.ravel(), pairwise_acc[np.tri(self.n_states, dtype=np.bool)]])

		# Return a RealVector
		psi_ret = RealVector()
		psi_ret.set(psi)

		dummy.io.message(MSG_DEBUG, '', 0, '[PSI] y =\n' + str(y_seq_arr) + '\n')
		dummy.io.message(MSG_DEBUG, '', 0, '[PSI] x =\n' + str(feat_vec.ravel()) + '\n')
		dummy.io.message(MSG_DEBUG, '', 0, '[PSI] psi =\n' + str(psi) + '\n')

		dummy.io.message(MSG_DEBUG, '', 0, 'Leaving '
						'GridCRFStructuredModel::get_joint_feature_vector.\n')

		return psi_ret

	def delta_loss(self, y1, y2):
		'''
		Hamming loss
		'''
		y1_seq_arr = Sequence.obtain_from_generic(y1).get_data()
		y2_seq_arr = Sequence.obtain_from_generic(y2).get_data()
		return np.float64(np.sum(y1_seq_arr != y2_seq_arr))

	def argmax(self, w, feat_idx, training):
		'''
		MAP-MRF inference
		'''
		dummy.io.message(MSG_DEBUG, '', 0, 'Entering GridCRFStructuredModel::argmax.\n')

		self._check_w_shape_(w)
		features = RealMatrixFeatures.obtain_from_generic(self.get_features())
		feat_vec = features.get_feature_vector(feat_idx)
		# Transpose to deal with the observations in row-major
		feat_vec = feat_vec.T
		self._check_feat_shape_(feat_vec)

		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] w =\n' + str(w.get()) + '\n')
		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] x =\n' + str(feat_vec.ravel()) + '\n')

		# Compute potentials
		unary_potentials = self._get_unary_potentials_(feat_vec, w)
		pairwise_potentials = self._get_pairwise_potentials_(w)

		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] unary_potentials =\n' + 
												str(unary_potentials) + '\n')
		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] pairwise_potentials =\n' + 
												str(pairwise_potentials) + '\n')

		# Make square grid edges
		edges = util.make_grid_edges(size=np.sqrt(feat_vec.shape[0]))

		if training:
			# Loss-augmented (LA) inference
			for l in np.arange(self.n_states):
				y_true_data = self.get_labels().get_label(feat_idx)
				y_true_seq_arr = Sequence.obtain_from_generic(y_true_data).get_data()
				unary_potentials[y_true_seq_arr != l, l] += 1.

			dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] GT for LA =\n' + 
												str(y_true_seq_arr) + '\n')
			dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] unary_potentials after LA=\n' + 
												str(unary_potentials) + '\n')

		# Inference by Linear Programming relaxation
		y_pred, energy = inference_methods.inference_lp(unary_potentials,
														pairwise_potentials, edges,
														return_energy=True)

		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] y_pred =\n' + str(y_pred) + '\n')
		dummy.io.message(MSG_DEBUG, '', 0, '[ARGMAX] energy = ' + str(energy) + '\n')

		res = ResultSet()
		res.score = energy
		y_pred_seq = Sequence(np.array(y_pred, dtype=np.int32))

		res.psi_pred = self.get_joint_feature_vector(feat_idx, y_pred_seq)
		res.argmax = Sequence(np.array(y_pred, dtype=np.int32))
		if training == True:
			res.delta = StructuredModel.delta_loss(self, feat_idx, y_pred_seq)
			res.psi_truth.set(StructuredModel.get_joint_feature_vector(self, feat_idx, feat_idx))

		dummy.io.message(MSG_DEBUG, '', 0, 'Leaving GridCRFStructuredModel::argmax.\n')
		return res

	def _check_feat_shape_(self, feat_vec, n_nodes=0):
		'''
		Check that the size of feat_vec is correct
		'''
		if n_nodes != 0 and feat_vec.shape[0] != n_nodes:
			raise ValueError("The observation should have %d nodes, it has %d instead."
							 % (n_nodes, feat_vec.shape[0]))

		if feat_vec.shape[1] != self.n_features:
			raise ValueError("The observation should have %d features per node, it has %d instead."
							 % (self.n_features, feat_vec.shape[1]))

	def _check_w_shape_(self, w):
		'''
		Check that the size of w is correct
		'''
		if w.vlen != self.dim:
			raise ValueError("Weight vector of wrong size. Expected %d, instead it has %s elements."
							 % (self.dim, w.vlen))

	def _get_unary_potentials_(self, feat_vec, w):
		'''
		Compute CRF unary potentials
		'''
		# Arrange the n_states times n_features first elements of w in matrix form
		unary_params = w[:self.n_states*self.n_features].reshape(self.n_states, self.n_features)
		return np.dot(feat_vec, unary_params.T)

	def _get_pairwise_potentials_(self, w):
		'''
		Compute CRF pairwise potentials
		'''
		if self.only_unaries:
			return np.zeros((self.n_states, self.n_states))
		else:
			# Get the last n_states*(n_states+1)/2 elements of w
			pairwise_flat = np.asarray(w[self.n_states*self.n_features:])
			# Arrange them in matrix form and force symmetry
			pairwise_params = np.zeros((self.n_states, self.n_states))
			pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
			return (pairwise_params + pairwise_params.T - np.diag(np.diag(pairwise_params)))

try:
	from shogun.Structure	import PrimalMosekSOSVM
except ImportError:
	print "recompile shogun with mosek"
	import sys
	sys.exit(0)

import toy_datasets			as toy
import matplotlib.pyplot	as plt
import pdb
import sys

from modshogun			import MSG_DEBUG, MSG_INFO
from shogun.Loss		import HingeLoss
from shogun.Structure	import StructuredAccuracy
from subgradient_sosvm	import SubgradientSOSVM, StochasticSubgradientSOSVM

Plot			= True
SaveFigs		= False
SaveLatexFigs	= False

dummy = HingeLoss()
if len(sys.argv)>1:
	if sys.argv[1]=='info':
		print 'Info mode set'
		dummy.io.set_loglevel(MSG_INFO)
	elif sys.argv[1]=='debug':
		print 'Debug mode set'
		dummy.io.set_loglevel(MSG_DEBUG)
	elif sys.argv[1]=='silent':
		print 'Silent mode set'
	else:
		print 'Option %s not available, silent mode activated.'

# Training data formed by n_samples examples
n_samples = 20
# Square grid CRFs of size times size
size = 12
# Observation of n_features dimension
n_features = 3 # has to matched with the one used by toy.generate_*
# Number of possible values that each node may take
n_states = 3 # has to matched with the one used by toy.generate_*
# Noise level
noise = 0.7

if len(sys.argv)>2 and sys.argv[2]=='load':
	import pickle
	X = pickle.load(open('X.p', 'rb'))
	Y = pickle.load(open('Y.p', 'rb'))
	n_states = 2
	n_features = 2
	assert(Y.shape[1] == Y.shape[2])
	size = Y.shape[1]
	n_samples = Y.shape[0]
else:
#   X, Y = toy.generate_gaussian_smoothed_pattern(n_samples, size=size)
#	X, Y = toy.generate_blocks(n_samples, n_rows=size, n_cols=size)
	X, Y = toy.generate_blocks_multinomial(n_samples, n_rows=size, n_cols=size, noise=noise)
#	X, Y = toy.generate_checker(n_samples, n_rows=size, n_cols=size)
### These ones below do not work fine
#	X, Y = toy.generate_square_with_hole(n_samples, total_size=size)
#	X, Y = toy.generate_checker_multinomial(n_samples, size=size)

# Some magic so SequenceLabels can be used even though the labels here are square matrices
#Y_flat = Y.T.reshape(size**2, n_samples).T.reshape(-1) ## col-major, old bug
Y_flat = Y.reshape(-1)	## row-major
labels = SequenceLabels(Y_flat, size**2, n_samples, n_states)

# More magic so RealMatrixFeatures can be used even though the features here are tensors
X_matl = np.zeros((n_features, 0))
for x in X:
	X_matl = np.concatenate((X_matl, x.reshape(-1, n_features).T), axis=1)
features = RealMatrixFeatures(X_matl, size**2, n_samples)

loss = HingeLoss()
pairwise_model = GridCRFStructuredModel(features, labels, False)
unary_model = GridCRFStructuredModel(features, labels, True)
#sosvm = PrimalMosekSOSVM(model, loss, labels)
#sosvm = SubgradientSOSVM(model, loss, labels, debug_at_iteration=10)
pairwise_sosvm = StochasticSubgradientSOSVM(pairwise_model, loss, labels, debug_at_iteration=10, max_iterations=100)
print 'Training with pairwise model'
pairwise_sosvm.train()
#dummy.io.message(MSG_DEBUG, '', 0, 'w =\n%s\n' % str(sosvm.get_w()))
unary_sosvm = StochasticSubgradientSOSVM(unary_model, loss, labels, debug_at_iteration=10, max_iterations=100)
print 'Training with unary model'
unary_sosvm.train()

pairwise_predicted = pairwise_sosvm.apply()
unary_predicted = unary_sosvm.apply()

evaluator = StructuredAccuracy()
pairwise_acc = evaluator.evaluate(pairwise_predicted, labels)
unary_acc = evaluator.evaluate(unary_predicted, labels)
print 'Pairwise accuracy ' + str(pairwise_acc)
print 'Unary accuracy ' + str(unary_acc)

'''
# Model learnt by PrimalMosekSOSVM externally, it should produce accuracy of about 0.99
sosvm.set_w(np.array([0, -5.9628, 0, 0, 8.1616e-2, -1.4641e1, 1.3547e-5]))
predicted = sosvm.apply()
acc = evaluator.evaluate(predicted, labels)
dummy.io.message(MSG_INFO, '', 0, 'Hardcoded model\'s accuracy: %.4f\n' % acc)
'''
if SaveFigs:
	for i in xrange(n_samples):
		fig, axarr = plt.subplots(1, 3)
		axarr[0].matshow(Y[i])
		axarr[0].set_title("GT")
		pred_seq = Sequence.obtain_from_generic(pairwise_predicted.get_label(i))
		axarr[1].matshow(pred_seq.get_data().reshape((size,size)))
		axarr[1].set_title("Prediction")
		pred_seq = Sequence.obtain_from_generic(unary_predicted.get_label(i))
		axarr[2].matshow(pred_seq.get_data().reshape((size,size)))
		axarr[2].set_title("Only unaries")
		fig.savefig("data_%03d.png" % i)

if Plot:
	f, axarr = plt.subplots(3,5)
	for i in xrange(5):
		axarr[0,i].matshow(Y[i,:,:])
		pred_seq = Sequence.obtain_from_generic(pairwise_predicted.get_label(i))
		axarr[1,i].matshow(pred_seq.get_data().reshape((size,size)))
		pred_seq = Sequence.obtain_from_generic(unary_predicted.get_label(i))
		axarr[2,i].matshow(pred_seq.get_data().reshape((size,size)))
	plt.show()

if SaveLatexFigs:
	for i in xrange(n_samples):
		pyplot.matshow(Y[i])
		pyplot.xticks([])
		pyplot.yticks([])
		pyplot.savefig('gt_%01d_noise_%03d.svg' % (round(10*noise), i), format='svg', bbox_inches=0)

		pred_seq = Sequence.obtain_from_generic(pairwise_predicted.get_label(i))
		pyplot.matshow(pred_seq.get_data().reshape((size,size)))
		pyplot.xticks([])
		pyplot.yticks([])
		pyplot.savefig("pairwise_%01d_noise_%03d.svg" % (round(10*noise), i), format='svg', bbox_inches=0)

		pred_seq = Sequence.obtain_from_generic(unary_predicted.get_label(i))
		pyplot.matshow(pred_seq.get_data().reshape((size,size)))
		pyplot.xticks([])
		pyplot.yticks([])
		pyplot.savefig("unary_%01d_noise_%03d.svg" % (round(10*noise), i), format='svg', bbox_inches=0)
