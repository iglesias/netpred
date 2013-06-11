import toy_datasets			as toy
import matplotlib.pyplot	as pyplot

from itertools import product

# Define constants
N = 5
size = 12
noise = 1.25

# Data generation
#X, Y = toy.generate_blocks_multinomial(n_samples=N, noise=noise, n_rows=size, n_cols=size)
X, Y = toy_data.generate_blocks_multinomial(n_samples=N, noise=noise, n_rows=size, n_cols=size)
#X, Y = toy.generate_checker_multinomial(n_samples=N, noise=noise)
#X, Y = toy.generate_square_with_hole(n_samples=N)
print X.shape
print Y.shape

# Visualization
fig, axarr = pyplot.subplots(X.shape[-1]+1,N)
for i, j in product(xrange(N), xrange(X.shape[-1])):
	axarr[0,i].matshow(Y[i,:,:])
	axarr[j+1,i].matshow(X[i,:,:,j])

pyplot.show()
