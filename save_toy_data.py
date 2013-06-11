import toy_datasets			as toy
import matplotlib.pyplot	as pyplot

from itertools import product

# Define constants
N = 1
size = 12
noise = 1.25

# Data generation
X, Y = toy.generate_blocks_multinomial(n_samples=N, noise=noise, n_rows=size, n_cols=size)

# Save files
pyplot.matshow(Y[0])
pyplot.xticks([])
pyplot.yticks([])
pyplot.savefig('figures/toy_label.svg', format='svg', bbox_inches=0)

pyplot.matshow(X[0,:,:,0])
pyplot.xticks([])
pyplot.yticks([])
pyplot.savefig('fugres/toy_feature_0.svg', format='svg', bbox_inches=0)

pyplot.matshow(X[0,:,:,1])
pyplot.xticks([])
pyplot.yticks([])
pyplot.savefig('figures/toy_feature_1.svg', format='svg', bbox_inches=0)
