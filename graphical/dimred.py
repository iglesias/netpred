# Based on dimred.py example from Shogun

"""
Shogun demo

Fernando J. Iglesias Garcia

This example shows the use of dimensionality reduction methods,
Stochastic Proximity Embedding (SPE) and Isomap. The data selected to
be embedded is an helix. The reduction achieved with Isomap is better,
more robust against noise. Isomap exploits the  parametrization of
the input data.
"""

import math
import mpl_toolkits.mplot3d as mpl3
import numpy as np
import pylab
import util

from shogun.Features  import RealFeatures
from shogun.Converter import StochasticProximityEmbedding, SPE_GLOBAL
from shogun.Converter import SPE_LOCAL, Isomap, DiffusionMaps
import latex_plot_inits

# Number of data points
N = 500

# Generate helix
t = np.linspace(1, N, N).T / N 
t = t*2*math.pi
X = np.r_[ [ ( 2 + np.cos(8*t) ) * np.cos(t) ],
           [ ( 2 + np.cos(8*t) ) * np.sin(t) ],
           [ np.sin(8*t) ] ]

# Bi-color helix
labels = np.round( (t*1.5) ) % 2

y1 = labels == 1
y2 = labels == 0

# Plot helix

fig = pylab.figure()
fig.add_subplot(1, 1, 1, projection = '3d')

pylab.plot(X[0, y1], X[1, y1], X[2, y1], 'rx')
pylab.plot(X[0, y2], X[1, y2], X[2, y2], 'go')

pylab.title('Original 3D Helix')
pylab.xlabel('x')
pylab.ylabel('y')

# Create features instance
features = RealFeatures(X)

# Create Stochastic Proximity Embedding converter instance
converter = StochasticProximityEmbedding()

# Set target dimensionality
converter.set_target_dim(2)
# Set strategy
converter.set_strategy(SPE_GLOBAL)

# Compute SPE embedding
embedding = converter.embed(features)

X = embedding.get_feature_matrix()

#fig.add_subplot(3, 1, 2)
fig = pylab.figure()

pylab.plot(X[0, y1], X[1, y1], 'rx')
pylab.plot(X[0, y2], X[1, y2], 'go')

pylab.title('Stochastic Proximity Embedding with global strategy')
pylab.xlabel('x')
pylab.ylabel('y')

# Compute Isomap embedding (for comparison)
converter = Isomap()
converter.set_target_dim(2)
converter.set_k(6)

embedding = converter.embed(features)

X = embedding.get_feature_matrix()

#fig.add_subplot(3, 1, 3)
fig = pylab.figure()

pylab.plot(X[0, y1], X[1, y1], 'rx')
pylab.plot(X[0, y2], X[1, y2], 'go')

pylab.title('Isomap')
pylab.xlabel('x')
pylab.ylabel('y')

pylab.connect('key_press_event', util.quit)
pylab.show()
