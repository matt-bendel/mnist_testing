import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

N = 1
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
Sigma = 2 * np.random.rand(N, N)
Sigma = np.dot(Sigma, Sigma.transpose())
Sigma[1, 1] = Sigma[0, 0]
cov = 0 #Sigma[0, 1]

U, S, Vh = np.linalg.svd(Sigma)
V = Vh.conj().T

var_1 = Sigma[0, 0]
var_2 = Sigma[1, 1]

sig_1 = np.sqrt(var_1)
sig_2 = np.sqrt(var_2)

Sig1_linspace = np.linspace(sig_1 - 0.8 * sig_1, sig_1 + 0.8 * sig_1, m)
Sig2_linspace = np.linspace(sig_2 - 0.8 * sig_2, sig_2 + 0.8 * sig_2, m)
Cov_linspace = np.linspace(cov - 0.8 * cov, cov + 0.8 * cov, m)

sig1_mesh, sig2_mesh = np.meshgrid(Sig1_linspace, Sig2_linspace, indexing='ij')

x = np.random.multivariate_normal(mu, Sigma, n)
x = np.expand_dims(x, 0).repeat(m, 0)
x = np.expand_dims(x, 0).repeat(m, 0)

x_hat = np.zeros((m, m, n, N, P))


for k, cov_val in enumerate(Cov_linspace):
    pca_reg_inner_vals = np.zeros((m, m))
    pca_reg_outer_vals = np.zeros((m, m))
    joint_reg = np.zeros((m, m))
    l1_vals = np.zeros((m, m))
    lamda_ks = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            Sig_samp = np.zeros((N, N))
            Sig_samp[0, 0] = sig1_mesh[i, j] ** 2
            Sig_samp[1, 1] = sig2_mesh[i, j] ** 2
            Sig_samp[0, 1] = cov_val
            Sig_samp[1, 0] = cov_val

            for z in range(P):
                x_hat[i, j, :, :, z] = np.random.multivariate_normal(mu, Sig_samp, n)

            pca_reg_inner_vals[i, j], pca_reg_outer_vals[i, j], _ = pca_reg(x_hat[i, j, :, :, :], mu, S, V)
            l1_vals[i, j] = l1_std_p(x[i, j, :, :], x_hat[i, j, :, :, :])
            joint_reg[i, j] = pca_reg_inner_vals[i, j] + l1_vals[i, j]

    plt.figure()
    plt.contour(sig1_mesh, sig2_mesh, pca_reg_inner_vals, 100)
    plt.colorbar()
    plt.plot(sig_1, sig_2, 'o')
    plt.savefig(f'contours/inner_expectation_sig1_v_sig2_cov={cov_val:.2f}.png')

    plt.figure()
    plt.contour(sig1_mesh, sig2_mesh, l1_vals, 100)
    plt.colorbar()
    plt.plot(sig_1, sig_2, 'o')
    plt.savefig(f'contours/l1_std_p_sig1_v_sig2_cov={cov_val:.2f}.png')

    plt.figure()
    plt.contour(sig1_mesh, sig2_mesh, joint_reg, 100)
    plt.colorbar()
    plt.plot(sig_1, sig_2, 'o')
    plt.savefig(f'contours/joint_reg_sig1_v_sig2_cov={cov_val:.2f}.png')
