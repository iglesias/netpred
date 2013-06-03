from shogun.Features	import RealMatrixFeatures
from shogun.Structure	import HMSVMModel, SMT_TWO_STATE, SequenceLabels, TwoStateModel, \
								Sequence

import numpy

class MLHMM:
	'''
	Hidden Markov Model (HMM) whose parameters are estimated in a Maximum
	Likelihood (ML) fashion from training data. Currently, it only supports
	two-state sequences.
	'''
	def __init__(self, model):
		# Setup of the HMM
		self.model = model
		features = RealMatrixFeatures.obtain_from_generic(model.get_features())
		labels = model.get_labels()
		assert(labels.get_num_labels() == features.get_num_vectors())
		self.num_states = 4
		self.num_free_states = 2
		self.num_features = features.get_num_features()
		self.num_plif_nodes = 20
		# Model parameters
		self.transition_scores = numpy.zeros(self.num_free_states**2)
		self.feature_scores = numpy.zeros((self.num_features, self.num_free_states,
				self.num_plif_nodes))

	'''
	def __init__(self, transition_scores, feature_scores, limits):
		# Setup of the HMM
		self.model = HMSVMModel(None, None, SMT_TWO_STATE, 0, True)
		self.num_states = 4
		self.num_free_states = feature_scores.shape[1]
		self.num_features = feature_scores.shape[0]
		self.num_plif_nodes = limits.shape[0]
		# Model parameters
		self.transition_scores = transition_scores
		self.feature_scores = feature_scores
	'''

	@property
	def transition_scores(self):
		return self.transition_scores

	@property
	def feature_scores(self):
		return self.feature_scores

	def train(self):
		'''
		Estimate model parameters from data.
		'''
		self.model.init_training()
		# Initialize frequency accumulators
		transmission_acc = numpy.zeros((self.num_states, self.num_states))
		emission_acc = numpy.zeros(self.num_states*self.num_features*self.num_plif_nodes)
		# Go through training data, counting state transitions and feature emissions
		assert(self.model.get_labels().get_num_labels() == 
				self.model.get_features().get_num_vectors())
		for i in xrange(self.model.get_labels().get_num_labels()):
			# Update transmission and emission counts internally in the model
			self.model.get_joint_feature_vector(i,i)
			# Update accumulators
			transmission_acc += self.model.get_transmission_weights()
			emission_acc += self.model.get_emission_weights()

		# Compute model parameters from the accumulators
		state_model = TwoStateModel()
		w = state_model.weights_to_vector(transmission_acc, emission_acc,
				self.num_features, self.num_plif_nodes)
##		print w
##		print '# transmission & emission weights = ', 
##				numpy.prod(transmission_acc.shape)+emission_acc.shape[0]
##		print '# params = ', num_free_states*(num_free_states + num_features*num_plif_nodes) 

		# Smooth counts equal to zero to avoid zero-division warnings
		w[w == 0.0] = 1e-100

		# Compute transition scores as transition frequencies, i.e. estimate
		# probabilities using the ML rule
		self.transition_scores[0] = numpy.log(w[0] / w[0:2].sum())
		self.transition_scores[1] = numpy.log(w[1] / w[0:2].sum())
		self.transition_scores[2] = numpy.log(w[2] / w[2:4].sum())
		self.transition_scores[3] = numpy.log(w[3] / w[2:4].sum())

		# Compute emission or feature scores as frequencies of occurrence
		# Offset given by the number of transition_scores, used to index w
		offset = self.transition_scores.shape[0]
		self.feature_scores = numpy.zeros((self.num_features, self.num_free_states,
				self.num_plif_nodes))
		for s in xrange(self.num_free_states):
			state_idx = s*self.num_features*self.num_plif_nodes 
			for f in xrange(self.num_features):
				feat_idx = f*self.num_plif_nodes
				idx = offset + state_idx + feat_idx
				self.feature_scores[f,s,:] = numpy.log(w[idx:idx+self.num_plif_nodes] 
						/ w[idx:idx+self.num_plif_nodes].sum())

	def apply(self, features=None):
		'''
		Perform prediction using the trained model.
		'''
		if features != None:
			self.model.set_features(features)
		else:
			features = self.model.get_features()

		prediction = SequenceLabels(features.get_num_vectors(), self.num_free_states)
		w = numpy.hstack([self.transition_scores, self.feature_scores[:,0,:].ravel(),
				self.feature_scores[:,1,:].ravel()])

		for i in xrange(features.get_num_vectors()):
			result = self.model.argmax(w, i, False)
			prediction.add_label(result.argmax)

		return prediction
