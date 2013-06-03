# Based on svm.py example from Shogun

import pylab
import numpy
import util
import latex_plot_inits

from shogun.Features	import *
from shogun.Classifier	import *
from shogun.Kernel		import *

util.set_title('SVM')
util.NUM_EXAMPLES=200

width = 5

# positive examples
pos = util.get_realdata(True)
pylab.plot(pos[0,:], pos[1,:], "rs")

# negative examples
neg = util.get_realdata(False)
pylab.plot(neg[0,:], neg[1,:], "bo")

# train svm
labels = util.get_labels()
train = util.get_realfeatures(pos, neg)
gk = GaussianKernel(train, train, width)
svm = LibSVM(10.0, gk, labels)
svm.train()

x, y, z = util.compute_output_plot_isolines(svm, gk, train)
pylab.contour(x, y, z, linewidths=1, colors='black', hold=True)
pylab.axis('tight')
pylab.title('Binary SVM classification with Gaussian kernel')
pylab.xlabel('x')
pylab.ylabel('y')

pylab.connect('key_press_event', util.quit)
pylab.show()

