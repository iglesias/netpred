import numpy as np

from shogun.Structure	import LinearStructuredOutputMachine

class SubgradientSOSVM(LinearStructuredOutputMachine):
	'''
	TODO DOC
	'''
	def __init__(self, model, loss, labels, max_iterations=1000, learning_rate=0.001, C=1.0):
		LinearStructuredOutputMachine.__init__(self)
		self.set_model(model)
		self.set_loss(loss)
		self.set_labels(labels)
		self.max_iterations = max_iterations
		self.learning_rate = learning_rate
		self.C = C

	'''
	TODO DOC
	'''
	def train(self):
		# Structured model
		model = self.get_model()
		# Dimensionality of the joint feature space
		M = model.get_dim()
		# Number of training examples
		N = model.get_features().get_num_vectors()

		# Initialize the weight vector
		w = np.zeros(M)

		for t in xrange(self.max_iterations):
			print 'Subgradient descent iteration #%d' % (t+1)

			# Initialize the cumulative subgradient
			subgrad = np.zeros(M)

			# Online learning
			for i in xrange(N):
				# Loss-augmented inference for training example i
				res = model.argmax(w, i, True)
				# Subgradient for training example i
				subgrad += res.psi_pred.get() - res.psi_truth.get()

			# Update weight vector
			w -= self.learning_rate*(w - 1.0*self.C/N*subgrad)

		self.set_w(w)

class StochasticSubgradientSOSVM(LinearStructuredOutputMachine):
	'''
	TODO DOC
	'''
	def __init__(self, model, loss, labels, max_iterations=1000, learning_rate=0.001, C=1.0):
		LinearStructuredOutputMachine.__init__(self)
		self.set_model(model)
		self.set_loss(loss)
		self.set_labels(labels)
		self.max_iterations = max_iterations
		self.learning_rate = learning_rate
		self.C = C

	'''
	TODO DOC
	'''
	def train(self):
		# Structured model
		model = self.get_model()
		# Dimensionality of the joint feature space
		M = model.get_dim()
		# Number of training examples
		N = model.get_features().get_num_vectors()

		# Initialize the weight vector
		w = np.zeros(M)

		for t in xrange(self.max_iterations):
			print 'Stochastic subgradient descent iteration #%d' % (t+1),

			# Select randomly a training example
			i = np.random.random_integers(0, N-1)
			print 'training example %d chosen' % i
			# Loss-augmented inference for training example i
			res = model.argmax(w, i, True)
			# Subgradient update
			subgrad = res.psi_pred.get() - res.psi_truth.get()
			w -= self.learning_rate*(w - 1.0*self.C/N*subgrad)

		self.set_w(w)
