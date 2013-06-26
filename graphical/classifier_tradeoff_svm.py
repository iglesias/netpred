#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import latex_plot_inits

parameter_list = [[100, 10, 1, 5e-3, 5], [100, 5, 1, 1, 100]]

def train_svm(feats_train, labels, C=1):
	from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC, L2R_L2LOSS_SVC_DUAL

	epsilon = 1e-3
	svm = LibLinear(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(False)
	svm.train()

	return svm

def plot_hyperplane(svm, x_min, x_max, col='k'):
	w = svm.get_w()
	
	v = w[0:-1]
	b = w[-1]

	hx = np.linspace(x_min-1,x_max+1)
	hy = -v[1]/v[0] * hx

	plt.plot(hx, -1/v[1]*(v[0]*hx+b), col, linewidth=2.0)

def classifier_non_separable_svm(n=100, distance=5, seed=None, C1=1, C2=100):
	'''
	n is the number of examples per class and m is the number of examples per class that gets its
	label swapped to force non-linear separability

	C1 and C2 are the two regularization values used
	'''
	from shogun.Features import RealFeatures, BinaryLabels

	# 2D data
	_DIM = 2

	# To get the nice message that the perceptron has converged
	dummy = BinaryLabels()

	np.random.seed(seed)

	# Produce some (probably) linearly separable training data by hand
	# Two Gaussians at a far enough distance
	X = np.array(2*np.random.randn(_DIM,n))+distance
	Y = np.array(1.5*np.random.randn(_DIM,n))-distance
	label_train_twoclass = np.hstack((np.ones(n), -np.ones(n)))
	# The last 5 points of the positive class are closer to the negative examples
	X[:,-3:-1] = np.random.randn(_DIM, 2) - 0.5*distance/2

	fm_train_real = np.hstack((X,Y))
	# add a feature with all ones to learn implicitily a bias
	fm_train_real = np.vstack((fm_train_real, np.ones(2*n)))
	feats_train = RealFeatures(fm_train_real)
	labels = BinaryLabels(label_train_twoclass)

	# Find limits for visualization
	x_min = min(np.min(X[0,:]), np.min(Y[0,:]))
	x_max = max(np.max(X[0,:]), np.max(Y[0,:]))

	# Train first SVM and plot its hyperplane
	svm1 = train_svm(feats_train, labels, C1)
	plot_hyperplane(svm1, x_min, x_max, 'g')

	# Train second SVM and plot its hyperplane
	svm2 = train_svm(feats_train, labels, C2)
	plot_hyperplane(svm2, x_min, x_max, 'y')

	# Plot the two-class data
	plt.scatter(X[0,:], X[1,:], s=40, marker='o', facecolors='none', edgecolors='b')
	plt.scatter(Y[0,:], Y[1,:], s=40, marker='s', facecolors='none', edgecolors='r')

	# Customize the plot

	y_min = min(np.min(X[1,:]), np.min(Y[1,:]))
	y_max = max(np.max(X[1,:]), np.max(Y[1,:]))

	plt.axis([x_min-1, x_max+1, y_min-1, y_max+1])
	plt.title('SVM trade-off')
	plt.xlabel('x')
	plt.ylabel('y')

	plt.show()

	return svm1, svm2 

if __name__=='__main__':
	print('SVM non-linearly separable data graphical')
	classifier_non_separable_svm(*parameter_list[0])
