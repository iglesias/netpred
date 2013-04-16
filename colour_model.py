#!/usr/bin/env python

import pdb
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def rgb2grey(rgb):
	r, g, b = np.rollaxis(rgb[...,:3], axis=-1)
	return 0.299*r + 0.587*g + 0.114*b

imgs = ['2008_002338','2009_001036','2008_002697','2008_002993','2010_005919']
fgpxs = np.zeros((3,0))
bgpxs = np.zeros((3,0))

for img in imgs:
	start = time.time()
	print '>>>> Processing %s' % img
	fname_labelimg = "SegmentationClass/%s.png" % img
	fname_trueimg = "JPEGImages/%s.jpg" % img

	# Transform the segmented image to 0/1 (BG/FG) image
	rgblabel = mpimg.imread(fname_labelimg)
	greylabel = rgb2grey(rgblabel)
	X,Y = np.nonzero(greylabel)
	greylabel = np.zeros(greylabel.shape, dtype=int)
	greylabel[X,Y] = 1

	rgbtrue = mpimg.imread(fname_trueimg)

	fgpx = rgbtrue[X,Y].T

	X,Y = np.nonzero(greylabel == 0)
	bgpx = rgbtrue[X,Y].T

	fgpxs = np.hstack((fgpxs, fgpx))
	bgpxs = np.hstack((bgpxs, bgpx))

	print '\ttook %.2f (s)' % (time.time()-start)

fgmu = np.mean(fgpxs, axis=1)
bgmu = np.mean(bgpxs, axis=1)
fgsigma = np.cov(fgpxs)
bgsigma = np.cov(bgpxs)

#print 'mu =', mu
#print 'sigma =\n', sigma

fgisigma = np.linalg.inv(fgsigma)
bgisigma = np.linalg.inv(bgsigma)

fig, axarr = plt.subplots(2,len(imgs))

for imgidx in xrange(len(imgs)):
	start = time.time()
	print '<<<< Plotting %s' % imgs[imgidx]

	fname_trueimg = "JPEGImages/%s.jpg" % imgs[imgidx]
	rgbtrue = mpimg.imread(fname_trueimg)

	'''
	Non-vectorized old version
	fgloglhood = np.zeros(rgbtrue.shape[:-1])
	bgloglhood = np.zeros(rgbtrue.shape[:-1])

	for i in xrange(rgbtrue.shape[0]):
		for j in xrange(rgbtrue.shape[1]):
			fgloglhood[i,j] = -0.5 * np.dot(np.dot((rgbtrue[i,j,:] - fgmu).T, fgisigma), rgbtrue[i,j,:] - fgmu)
			bgloglhood[i,j] = -0.5 * np.dot(np.dot((rgbtrue[i,j,:] - bgmu).T, bgisigma), rgbtrue[i,j,:] - bgmu)
	'''

	# Compute Gaussian likelihood of the pixel being in the FG
	fgloglhood = np.sum(np.dot(rgbtrue, fgisigma) * rgbtrue, axis=2) + np.dot(np.dot(fgmu.T, fgisigma), fgmu) - 2*np.dot(np.dot(rgbtrue, fgisigma), fgmu)
	fgloglhood = -0.5*fgloglhood

	# Compute Gaussian likelihood of the pixel being in the BG
	bgloglhood = np.sum(np.dot(rgbtrue, bgisigma) * rgbtrue, axis=2) + np.dot(np.dot(bgmu.T, bgisigma), bgmu) - 2*np.dot(np.dot(rgbtrue, bgisigma), bgmu)
	bgloglhood = -0.5*bgloglhood

	axarr[0,imgidx].imshow(fgloglhood, cmap = plt.get_cmap('gray'))
	axarr[1,imgidx].imshow(bgloglhood, cmap = plt.get_cmap('gray'))

	print '\ttook %.2f (s)' % (time.time()-start)

plt.show()
