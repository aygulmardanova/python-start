from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

X = make_blobs(n_samples=3000, centers=4, n_features=1, center_box=(-20.0, 20.0))
print X

plt.plot(X[0], X[1], 'ro')
plt.axis([-20, 20, -1, 5])

plt.show()
