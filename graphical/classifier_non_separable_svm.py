#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import latex_plot_inits

parameter_list = [[20, 3, 5, None], [100, 10, 5, None]]

def classifier_non_separable_svm(n=100, m=10, distance=5, seed=None):
	'''
	n is the number of examples per class and m is the number of examples per class that gets its
	label swapped to force non-linear separability
	'''
	from shogun.Features import RealFeatures, BinaryLabels
	from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC

	# 2D data
	_DIM = 2

	# To get the nice message that the perceptron has converged
	dummy = BinaryLabels()

	np.random.seed(seed)

	# Produce some (probably) linearly separable training data by hand
	# Two Gaussians at a far enough distance
	X = np.array(np.random.randn(_DIM,n))+distance
	Y = np.array(np.random.randn(_DIM,n))
	# The last five points of each class are swapped to force non-linear separable data
	label_train_twoclass = np.hstack((np.ones(n-m), -np.ones(m), -np.ones(n-m), np.ones(m)))

	fm_train_real = np.hstack((X,Y))
	feats_train = RealFeatures(fm_train_real)
	labels = BinaryLabels(label_train_twoclass)


	# Train linear SVM
	C = 1
	epsilon = 1e-3
	svm = LibLinear(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(True)
	svm.train()

	# Get hyperplane parameters
	b = svm.get_bias()
	w = svm.get_w()

	# Find limits for visualization
	x_min = min(np.min(X[0,:]), np.min(Y[0,:]))
	x_max = max(np.max(X[0,:]), np.max(Y[0,:]))

	y_min = min(np.min(X[1,:]), np.min(Y[1,:]))
	y_max = max(np.max(X[1,:]), np.max(Y[1,:]))

	hx = np.linspace(x_min-1,x_max+1)
	hy = -w[1]/w[0] * hx

	plt.plot(hx, -1/w[1]*(w[0]*hx+b), 'k', linewidth=2.0)

	# Plot the two-class data
	pos_idxs = label_train_twoclass == +1;
	plt.scatter(fm_train_real[0, pos_idxs], fm_train_real[1, pos_idxs], s=40, marker='o', facecolors='none', edgecolors='b')

	neg_idxs = label_train_twoclass == -1;
	plt.scatter(fm_train_real[0, neg_idxs], fm_train_real[1, neg_idxs], s=40, marker='s', facecolors='none', edgecolors='r')

	# Customize the plot
	plt.axis([x_min-1, x_max+1, y_min-1, y_max+1])
	plt.title('SVM with non-linearly separable data')
	plt.xlabel('x')
	plt.ylabel('y')

	plt.show()

	return svm 

if __name__=='__main__':
	print('SVM non-linearly separable data graphical')
	classifier_non_separable_svm(*parameter_list[0])
