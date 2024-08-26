from sklearn.neighbors import NearestNeighbors
import torch


def circle(n, r=1):
    t = torch.linspace(0, 2 * 3.1415, n)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)


def parameterized_curve(f):
    def helper(n):
        return torch.stack(
            [torch.tensor(f(t)) for t in torch.linspace(0, 1, n)], 0
        )

    return helper


ß
spiral = parameterized_curve(
    lambda t: (t * torch.cos(2 * 3.1415 * t), t * torch.sin(2 * 3.1415 * t))
)
hyperbola = parameterized_curve(
    lambda t: [-1 + torch.cosh(4 * t) / 4, -1 + 1 * torch.sinh(2 * t)]
)

M, N = 10, 25
K = 1

# X = torch.randn(M, 2)
# Y = torch.randn(N, 2)

a, b = -2, 2
c, d = -2, 2

x = torch.linspace(a, b, M)
y = torch.linspace(c, d, M)
X = torch.stack(torch.meshgrid(x, y), 2).view(-1, 2)
# Y = circle(N, 1)
# Y = spiral(N)
Y = hyperbola(N)


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
