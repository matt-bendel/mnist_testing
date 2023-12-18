import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

N = 1
n = 100000
m = 5
P = 2


def l1_std_p(x, x_hat):
    beta = np.sqrt(2 / np.pi / P / (P + 1))
    return np.mean(np.abs(x - np.mean(x_hat, axis=-1)) - beta * np.sqrt(np.pi * P / (2 * (P - 1))) * np.std(x_hat, axis=-1))

mu = np.random.randn(N)
Sigma = 2 * np.random.rand(N, N)

if N > 1:
    Sigma = np.dot(Sigma, Sigma.transpose())
    Sigma[1, 1] = Sigma[0, 0]
    cov = 0 #Sigma[0, 1]
else:
    Sigma = Sigma[0, 0]

Mu_linspace = np.linspace(mu - 2, mu + 2, 5)
Sig_linspace = np.linspace(Sigma - 0.8 * Sigma, Sigma + 0.8 * Sigma, m)

mu_mesh, sig_mesh = np.meshgrid(Mu_linspace, Sig_linspace, indexing='ij')

x = np.random.normal(mu, Sigma, n)
x = np.expand_dims(x, 0).repeat(m, 0)
x = np.expand_dims(x, 0).repeat(m, 0)

x_hat = np.zeros((m, m, n, P))
l1_std_p_vals = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        mu_samp = mu_mesh[i, j]
        sig_samp = sig_mesh[i, j]

        for z in range(P):
            x_hat[i, j, :, z] = mu_samp + sig_samp * np.random.randn(n)

        l1_std_p_vals[i, j] = l1_std_p(x[i, j, :], x_hat[i, j, :, :])

plt.figure()
plt.contour(mu_mesh, sig_mesh, l1_std_p_vals, 50)
plt.colorbar()
plt.ylabel('sigma')
plt.xlabel('mu')
plt.plot(mu, Sigma, 'o')
plt.savefig(f'contours/scalar_test_l1_std_p.png')