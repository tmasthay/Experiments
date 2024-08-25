from sklearn.neighbors import NearestNeighbors
import torch

M, N = 100, 10
K = 4

X = torch.randn(M, 2)
Y = torch.randn(N, 2)

neigh = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
neigh.fit(X)
distances, indices = neigh.kneighbors(Y)

# visualize this with scatter plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c='blue', label='X')
plt.scatter(Y[:, 0], Y[:, 1], c='red', label='Y')
for i in range(N):
    for j in range(K):
        plt.plot(
            [Y[i, 0], X[indices[i, j], 0]], [Y[i, 1], X[indices[i, j], 1]], 'k-'
        )
plt.legend()
plt.savefig('scatter.png')
