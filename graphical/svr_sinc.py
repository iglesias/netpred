# Based on svr_sinc.py example from Shogun

import pylab 
import util
import latex_plot_inits

from shogun.Features	import *
from shogun.Regression	import *
from shogun.Kernel		import *

# Generate data
X, Y = util.get_sinedata()
# Create labels and features from data
feat = RealFeatures(X)
lab = RegressionLabels(Y.flatten())

# Constants used for regression
C = 10
width = 0.5
epsilon = 0.01

# Create kernel and SVR
gk = GaussianKernel(feat,feat, width)
svr = LibSVR(C, epsilon, gk, lab)

# Train machine
svr.train()

# Plot results
pylab.scatter(X, Y, marker='o', facecolors='r')
pylab.plot(X[0], svr.apply().get_labels(), linewidth=2)

pylab.axis('tight')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.title('Suppor Vector Regression with Gaussian kernel of data distributed in the form of a sinc')

pylab.connect('key_press_event', util.quit)
pylab.show()
