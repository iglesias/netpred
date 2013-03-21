import pickle
import toy_datasets as toy

size = 2
n_samples = 5

X, Y = toy.generate_blocks(n_samples, n_rows=size, n_cols=size)
print '%d samples of %dx%d elements generated.' % (n_samples, size, size)

pickle.dump(X, open('X.p', 'wb'))
pickle.dump(Y, open('Y.p', 'wb'))
