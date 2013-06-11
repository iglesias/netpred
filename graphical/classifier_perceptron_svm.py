#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import latex_plot_inits

parameter_list = [[20, 5, 1., 1000, 1, None, 5], [100, 5, 1., 1000, 1, None, 10]]

def min_distance(w, b, feats):
	'''
	Compute the minimum distance from the hyperplane to the dataset formed by feats
	'''
	min_dist = float('inf')
	for j in xrange(feats.get_num_vectors()):
		x = feats.get_feature_vector(j)
		dist = 1./np.linalg.norm(w) * ( np.dot(w, x) + b )
		# 1/||w|| (w^T x + b) is *signed* distance
		dist = np.abs(dist)

		if dist < min_dist:
			min_dist= dist

	return min_dist


def classifier_perceptron_graphical(n=100, distance=5, learn_rate=1., max_iter=1000, num_threads=1, seed=None, nperceptrons=5):
	from shogun.Features import RealFeatures, BinaryLabels
	from shogun.Classifier import Perceptron, LibLinear, L2R_L2LOSS_SVC
	from modshogun import MSG_INFO

	# 2D data
	_DIM = 2

	# To get the nice message that the perceptron has converged
	dummy = BinaryLabels()
#	dummy.io.set_loglevel(MSG_INFO)

	np.random.seed(seed)

	# Produce some (probably) linearly separable training data by hand
	# Two Gaussians at a far enough distance
	X = np.array(np.random.randn(_DIM,n))+distance
	Y = np.array(np.random.randn(_DIM,n))
	label_train_twoclass = np.hstack((np.ones(n), -np.ones(n)))

	fm_train_real = np.hstack((X,Y))
	feats_train = RealFeatures(fm_train_real)
	labels = BinaryLabels(label_train_twoclass)

	perceptron = Perceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	perceptron.set_initialize_hyperplane(False)

	# Find limits for visualization
	x_min = min(np.min(X[0,:]), np.min(Y[0,:]))
	x_max = max(np.max(X[0,:]), np.max(Y[0,:]))

	y_min = min(np.min(X[1,:]), np.min(Y[1,:]))
	y_max = max(np.max(X[1,:]), np.max(Y[1,:]))

	fig1, axes1 = plt.subplots(1,1)
	fig2, axes2 = plt.subplots(1,1)

	for i in xrange(nperceptrons):
		# Initialize randomly weight vector and bias
		perceptron.set_w(np.random.random(2))
		perceptron.set_bias(np.random.random())

		# Run the perceptron algorithm
		perceptron.train()

		# Construct the hyperplane for visualization
		# Equation of the decision boundary is w^T x + b = 0
		b = perceptron.get_bias()
		w = perceptron.get_w()

		hx = np.linspace(x_min-1,x_max+1)
		hy = -w[1]/w[0] * hx

		axes1.plot(hx, -1/w[1]*(w[0]*hx+b))
		axes2.plot(hx, -1/w[1]*(w[0]*hx+b), alpha=0.5)

		print('minimum distance with perceptron is %f' % min_distance(w, b, feats_train))

	C = 1
	epsilon = 1e-3
	svm = LibLinear(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(True)
	svm.train()

	b = svm.get_bias()
	w = svm.get_w()

	print('minimum distance with svm is        %f' % min_distance(w, b, feats_train))

	hx = np.linspace(x_min-1,x_max+1)
	hy = -w[1]/w[0] * hx

	axes2.plot(hx, -1/w[1]*(w[0]*hx+b), 'k', linewidth=2.0)

	# Plot the two-class data
	axes1.scatter(X[0,:], X[1,:], s=40, marker='o', facecolors='none', edgecolors='b')
	axes1.scatter(Y[0,:], Y[1,:], s=40, marker='s', facecolors='none', edgecolors='r')

	axes2.scatter(X[0,:], X[1,:], s=40, marker='o', facecolors='none', edgecolors='b')
	axes2.scatter(Y[0,:], Y[1,:], s=40, marker='s', facecolors='none', edgecolors='r')

	# Customize the plot
	axes1.axis([x_min-1, x_max+1, y_min-1, y_max+1])
	axes1.set_title('Rosenblatt\'s Perceptron Algorithm')
	axes1.set_xlabel('x')
	axes1.set_ylabel('y')

	axes2.axis([x_min-1, x_max+1, y_min-1, y_max+1])
	axes2.set_title('Support Vector Machine')
	axes2.set_xlabel('x')
	axes2.set_ylabel('y')

	plt.show()

	return perceptron

if __name__=='__main__':
	print('Perceptron graphical')
	classifier_perceptron_graphical(*parameter_list[0])
