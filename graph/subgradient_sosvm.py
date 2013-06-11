import numpy as np

from shogun.Structure	import LinearStructuredOutputMachine, StructuredAccuracy

class SubgradientSOSVM(LinearStructuredOutputMachine):
	'''
	TODO DOC
	'''
	def __init__(self, model, loss, labels, max_iterations=100, learning_rate=0.01, C=1.0, debug_at_iteration=101):
		LinearStructuredOutputMachine.__init__(self)
		self.set_model(model)
		self.set_loss(loss)
		self.set_labels(labels)
		self.max_iterations = max_iterations
		self.learning_rate = learning_rate
		self.C = C
		self.debug_at_iteration = debug_at_iteration

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

		try:
			for t in xrange(self.max_iterations):
				print 'Subgradient descent iteration #%d' % (t+1)

				# Initialize the cumulative subgradient
				subgrad = np.zeros(M)

				for i in xrange(N):
					# Loss-augmented inference for training example i
					res = model.argmax(w, i, True)
					# Subgradient for training example i
					subgrad = res.psi_truth.get() - res.psi_pred.get() - self.C*w
					# Update weight vector
					w += self.learning_rate*subgrad

				if (t+1) % self.debug_at_iteration == 0:
					# Track the primal objective
					obj = 0.
					self.set_w(w)
					for i in xrange(N):
						res = model.argmax(w, i, True)
						obj += max(0.0, res.delta - np.dot(w, res.psi_truth.get() - res.psi_pred.get()))
					obj += self.C * np.sum(w**2)
					print '\t\t\tPrimal objective: %.4f' % obj
		except KeyboardInterrupt:
			print '\t\t==== Execution stopped by user ===='

		self.set_w(w)

class StochasticSubgradientSOSVM(LinearStructuredOutputMachine):
	'''
	TODO DOC
	'''
	def __init__(self, model, loss, labels, max_iterations=1000, learning_rate=0.01, C=1.0, debug_at_iteration=1001):
		LinearStructuredOutputMachine.__init__(self)
		self.set_model(model)
		self.set_loss(loss)
		self.set_labels(labels)
		self.max_iterations = max_iterations
		self.learning_rate = learning_rate
		self.C = C
		self.debug_at_iteration = debug_at_iteration

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

		# Catch user interruption
		try:
			for t in xrange(self.max_iterations):
				print 'Stochastic subgradient descent iteration #%d' % (t+1),

				# Select randomly a training example
				i = np.random.random_integers(0, N-1)
				print 'training example %d chosen' % i
				# Loss-augmented inference for training example i
				res = model.argmax(w, i, True)
				# Subgradient update
				subgrad = res.psi_truth.get() - res.psi_pred.get() - self.C*w
				# Update weight vector
				w += self.learning_rate*subgrad
				self.set_w(w)

				if (t+1) % self.debug_at_iteration == 0:
					# Track the primal objective
					obj= 0.
					for i in xrange(N):
						res = model.argmax(w, i, True)
						obj += max(0., res.delta - np.dot(w, res.psi_truth.get() - res.psi_pred.get()))
					obj += 1/2. * self.C * np.sum(w**2)
					print '\t\t\tPrimal objective: %.4f' % obj

		except KeyboardInterrupt:
			print '\t\t==== Execution stopped by user ===='
