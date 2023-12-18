import numpy as np
import matplotlib.pyplot as plt

N = 2
n = 100000
m = 5
P = 2


def l1_std_p(x, x_hat):
    beta = np.sqrt(2 / (np.pi * P * (P + 1)))
    return np.mean(np.sum(np.abs(x - np.mean(x_hat, axis=-1)) - beta * np.std(x_hat, axis=-1), axis=-1))


def pca_reg(x_hat, mu, evals, evecs):
    outer_exp = 0
    inner_exp = 0
    lamda_ks = np.zeros(2)

    for k in range(evecs.shape[1]):
        eval_hat = np.zeros(n)

        for z in range(x_hat.shape[-1]):
            eval_hat += 1 / P * (np.matmul(x_hat[:, :, z] - mu[None, :], evecs[:, k]))**2

        lamda_ks[k] = np.mean(eval_hat)

        inner_exp += (evals[k] - np.mean(eval_hat)) ** 2
        outer_exp += np.mean((evals[k] - eval_hat) ** 2)

    return inner_exp, outer_exp, lamda_ks[0]


mu = np.random.randn(N)
Sigma = 7 * np.random.rand(N, N)
Sigma = np.dot(Sigma, Sigma.transpose())
Sigma[0, 1] = 0.5
Sigma[1, 0] = 0.5

U, S, Vh = np.linalg.svd(Sigma)
V = Vh.conj().T

var_1 = Sigma[0, 0]
var_2 = Sigma[1, 1]

sig_1 = np.sqrt(var_1)
sig_2 = np.sqrt(var_2)
cov = Sigma[0, 1]

Mu1_linspace = np.linspace(mu[0] - 2, mu[0] + 2, m)
Mu2_linspace = np.linspace(mu[1] - 2, mu[1] + 2, m)

Sig1_linspace = np.linspace(sig_1 - 0.5 * sig_1, sig_1 + 0.5 * sig_1, m)
Sig2_linspace = np.linspace(sig_2 - 0.5 * sig_2, sig_2 + 0.5 * sig_2, m)
Cov_linspace = np.linspace(cov - 0.5 * cov, cov + 0.5 * cov, m)

# mu1_mesh, sig1_mesh = np.meshgrid(Mu1_linspace, Sig1_linspace)
# mu2_mesh, sig2_mesh = np.meshgrid(Mu2_linspace, Sig2_linspace)
sig1_mesh, sig2_mesh = np.meshgrid(Sig1_linspace, Sig2_linspace, indexing='ij')
sig_mesh, cov_mesh = np.meshgrid(Sig1_linspace, Cov_linspace, indexing='ij')

x = np.random.multivariate_normal(mu, Sigma, n)
x = np.expand_dims(x, 0).repeat(m, 0)
x = np.expand_dims(x, 0).repeat(m, 0)

x_hat = np.zeros((m, m, n, N, P))
l1_std_p_vals = np.zeros((m, m))
pca_reg_vals_inner = np.zeros((m, m))
pca_reg_vals_outer = np.zeros((m, m))
pca_reg_vals_inner_cov = np.zeros((m, m))
pca_reg_vals_outer_cov = np.zeros((m, m))

lamda_ks = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        mu_samp = mu # np.array([mu1_mesh[i, j], mu2_mesh[i, j]])
        Sig_samp = np.zeros((N, N))
        Sig_samp[0, 0] = sig1_mesh[i, j] ** 2
        Sig_samp[1, 1] = sig2_mesh[i, j] ** 2
        Sig_samp[0, 1] = Sigma[0, 1]
        Sig_samp[1, 0] = Sigma[1, 0]

        for z in range(P):
            x_hat[i, j, :, :, z] = np.random.multivariate_normal(mu_samp, Sig_samp, n)

        l1_std_p_vals[i, j] = l1_std_p(x[i, j, :, :], x_hat[i, j, :, :, :])
        pca_reg_vals_inner[i, j], pca_reg_vals_outer[i, j], lamda_ks[i, j] = pca_reg(x_hat[i, j, :, :, :], mu_samp, S, V)

for i in range(m):
    for j in range(m):
        mu_samp = mu # np.array([mu1_mesh[i, j], mu2_mesh[i, j]])
        Sig_samp = np.zeros((N, N))
        Sig_samp[0, 0] = sig_mesh[i, j] ** 2
        Sig_samp[1, 1] = sig_mesh[i, j] ** 2
        Sig_samp[0, 1] = cov_mesh[i, j]
        Sig_samp[1, 0] = cov_mesh[i, j]

        for z in range(P):
            x_hat[i, j, :, :, z] = np.random.multivariate_normal(mu_samp, Sig_samp, n)

        pca_reg_vals_inner_cov[i, j], pca_reg_vals_outer_cov[i, j], _ = pca_reg(x_hat[i, j, :, :, :], mu_samp, S, V)

plt.figure()
plt.contour(sig1_mesh, sig2_mesh, pca_reg_vals_inner)
plt.plot(sig_1, sig_2, 'o')
plt.savefig('sig1_sig2_countour_inner.png')
plt.close()

plt.figure()
plt.contour(sig1_mesh, sig2_mesh, pca_reg_vals_outer)
plt.plot(sig_1, sig_2, 'o')
plt.savefig('sig1_sig2_countour_outer.png')
plt.close()

plt.figure()
plt.contour(sig_mesh, cov_mesh, pca_reg_vals_inner_cov)
plt.plot(sig_1, cov, 'o')
plt.savefig('sig1_sig2_countour_inner_cov.png')
plt.close()

plt.figure()
plt.contour(sig_mesh, cov_mesh, pca_reg_vals_outer_cov)
plt.plot(sig_1, cov, 'o')
plt.savefig('sig1_sig2_countour_outer_cov.png')
plt.close()