

from numpy import ones,zeros,cos,sin,concatenate
from numpy.random import randn

import pylab
import latex_plot_inits
import util

from modshogun import *

k =3
num = 1000
iter = 50000
dist =2.2

traindat=concatenate((concatenate((randn(1,num), randn(1,num)+dist, randn(1,num)+2*dist),1), concatenate((randn(1,num), randn(1,num)+dist, randn(1,num)-dist),1)),0)

trainlab=concatenate((ones(num), 2*ones(num), 3*ones(num)))

feats_train=RealFeatures(traindat)
distance=EuclideanDistance(feats_train, feats_train)
kmeans=KMeans(k, distance)
kmeans.train()

centers = kmeans.get_cluster_centers()
radi=kmeans.get_radiuses()

pylab.figure()
pylab.clf()
pylab.plot(traindat[0,trainlab==+1], traindat[1,trainlab==+1],'ro')
pylab.plot(traindat[0,trainlab==+2], traindat[1,trainlab==+2],'bs')
pylab.plot(traindat[0,trainlab==+3], traindat[1,trainlab==+3],'gx', )

plot(centers[0,:], centers[1,:], 'ko', hold=True)

for i in xrange(k):
	t = linspace(0, 2*pi, 100)
	pylab.plot(radi[i]*cos(t)+centers[0,i],radi[i]*sin(t)+centers[1,i],'k-', hold=True)

pylab.axis('equal')
pylab.title('Clustering with kMeans')
pylab.xlabel('x')
pylab.ylabel('y')

pylab.connect('key_press_event', util.quit)
pylab.show()
