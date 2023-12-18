import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
import matplotlib.mlab as mlab
from matplotlib.contour import QuadContourSet
from matplotlib.widgets import Slider

# Choose variances
# Try normalized cross covariance -1 -> 1
# Rotate samples by eigenvectors

N = 2
n = 500000
m = 5
P = 2
scale = 5
K = 2

def l1_std_p(x, x_hat):
    beta = np.sqrt(2 / np.pi / P / (P + 1))
    x_hat_zm = x_hat - np.mean(x_hat, axis=-1)[:, :, None]
    std_dev_est = np.sqrt(np.pi * P / (2 * (P - 1))) * np.mean(np.abs(x_hat_zm), axis=-1)
    return np.mean(np.sum(np.abs(x - np.mean(x_hat, axis=-1)) - beta * std_dev_est, axis=-1))

def l1_std_p_thry(sig_true, sig_fake):
    beta = np.sqrt(2 / np.pi / P / (P + 1))
    return np.sqrt(2 * (sig_true ** 2 + sig_fake ** 2 / P) / np.pi) - beta * sig_fake

def pca_reg(x_hat, mu, evals, evecs):
    outer_exp = 0
    inner_exp = 0

    for k in range(evecs.shape[1]):
        eval_hat = np.zeros(n)

        for z in range(x_hat.shape[-1]):
            eval_hat += 1 / (P - 1) * (np.matmul(x_hat[:, :, z] - np.mean(x_hat, axis=-1), evecs[:, k]))**2

        inner_exp += 1 / (K * evals[k] ** 2) * (evals[k] - np.mean(eval_hat)) ** 2

        # print(f'TRUE: {evals[k]}; FAKE: {np.mean(eval_hat)}')
        outer_exp += np.mean((evals[k] - eval_hat) ** 2)

    return inner_exp, outer_exp

def pca_reg_theory(Sig, evals, evecs):
    pca_val = 0

    for k in range(evecs.shape[1]):
        eval_hat = np.matmul(evecs[:, k].T, np.matmul(Sig, evecs[:, k]))

        pca_val += 1 / (K * evals[k] ** 2) * (evals[k] - eval_hat) ** 2

    return pca_val

mu = np.random.randn(N)
Sigma = scale * np.random.rand(N, N)


if N > 1:
    Sigma = np.dot(Sigma, Sigma.transpose()) + np.eye(N)
    Sigma[0, 0] = 9
    Sigma[1, 1] = 7
else:
    Sigma = Sigma[0, 0]

########### VECTOR CASE ##############
var_1 = Sigma[0, 0]
var_2 = Sigma[1, 1]

sig_1 = np.sqrt(var_1)
sig_2 = np.sqrt(var_2)

norm_cov = 0.5

Sigma[1, 0] = norm_cov * sig_1 * sig_2
Sigma[0, 1] = norm_cov * sig_1 * sig_2

Sig1_linspace = np.linspace(sig_1 - 0.5 * sig_1, sig_1 + 0.5 * sig_1, m)
Sig2_linspace = np.linspace(sig_2 - 0.5 * sig_2, sig_2 + 0.5 * sig_2, m)
Cov_linspace = np.linspace(0, 1, m)
sig1_mesh, sig2_mesh, cov_mesh = np.meshgrid(Sig1_linspace, Sig2_linspace, Cov_linspace, indexing='ij')

x = np.random.multivariate_normal(mu, Sigma, n)
x = np.expand_dims(x, 0).repeat(m, 0)
x = np.expand_dims(x, 0).repeat(m, 0)
x = np.expand_dims(x, 0).repeat(m, 0)

U, S, Vh = np.linalg.svd(Sigma)
V = Vh.conj().T

###### VARYING COV ########
pca_reg_inner_vals = np.zeros((m, m, m))
pca_theory_vals = np.zeros((m, m, m))
joint_reg = np.zeros((m, m, m))
l1_vals = np.zeros((m, m, m))
l1_vals_thry = np.zeros((m, m, m))

x_hat = np.zeros((m, m, m, n, N, P))

normalized_cov = cov_mesh
for i in range(m):
    print(f"#########{i}########")
    for j in range(m):
        for k in range(m):
            normalized_cov[i, j, k] = cov_mesh[i, j, k]
            sigs_true = [sig_1, sig_2]
            Sig_samp = np.zeros((N, N))
            Sig_samp[0, 0] = sig1_mesh[i, j, k] ** 2
            Sig_samp[1, 1] = sig2_mesh[i, j, k] ** 2
            Sig_samp[0, 1] = cov_mesh[i, j, k] * sig1_mesh[i, j, k] * sig2_mesh[i, j, k]
            Sig_samp[1, 0] = cov_mesh[i, j, k] * sig1_mesh[i, j, k] * sig2_mesh[i, j, k]

            for z in range(P):
                x_hat[i, j, k, :, :, z] = np.random.multivariate_normal(mu, Sig_samp, n)

            pca_reg_inner_vals[i, j, k], _ = pca_reg(x_hat[i, j, k, :, :, :], mu, S, V)
            l1_vals[i, j, k] = l1_std_p(x[i, j, k, :, :], x_hat[i, j, k, :, :, :])

            l1_val_thry = 0
            for z in range(N):
                l1_val_thry += l1_std_p_thry(sigs_true[z], np.sqrt(Sig_samp[z, z]))

            l1_vals_thry[i, j, k] = l1_val_thry
            joint_reg[i, j, k] = pca_reg_inner_vals[i, j, k] + l1_vals[i, j, k]
            pca_theory_vals[i, j, k] = pca_reg_theory(Sig_samp, S, V)

# sig_1 v. cov for each sig2




fig_sig1_v_cov, axs_sig1_v_cov = plt.subplots(4, m)

for i in range(m):
    sig_plt = sig1_mesh[:, i, :]
    cov_plt = normalized_cov[:, i, :]
    axs_sig1_v_cov[0, i].contour(sig_plt, cov_plt, pca_reg_inner_vals[:, i, :], 20)
    if i == 2:
        axs_sig1_v_cov[0, i].set_title(f'sig2 = {Sig2_linspace[i]}')
    if i == 0:
        axs_sig1_v_cov[0, i].set_ylabel('pca_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[0, i].plot(sig_1, norm_cov, 'o')
    axs_sig1_v_cov[0, i].set_yticklabels([])
    axs_sig1_v_cov[0, i].set_xticklabels([])
    axs_sig1_v_cov[0, i].set_xticks([])
    axs_sig1_v_cov[0, i].set_yticks([])

    axs_sig1_v_cov[1, i].contour(sig_plt, cov_plt, pca_theory_vals[:, i, :], 20)
    if i == 0:
        axs_sig1_v_cov[1, i].set_ylabel('pca_reg_theory')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[1, i].plot(sig_1, norm_cov, 'o')
    axs_sig1_v_cov[1, i].set_yticklabels([])
    axs_sig1_v_cov[1, i].set_xticklabels([])
    axs_sig1_v_cov[1, i].set_xticks([])
    axs_sig1_v_cov[1, i].set_yticks([])

    axs_sig1_v_cov[2, i].contour(sig_plt, cov_plt, l1_vals[:, i, :], 20)
    if i == 0:
        axs_sig1_v_cov[2, i].set_ylabel('l1_std')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[2, i].plot(sig_1, norm_cov, 'o')
    axs_sig1_v_cov[2, i].set_yticklabels([])
    axs_sig1_v_cov[2, i].set_xticklabels([])
    axs_sig1_v_cov[2, i].set_xticks([])
    axs_sig1_v_cov[2, i].set_yticks([])

    axs_sig1_v_cov[3, i].contour(sig_plt, cov_plt, joint_reg[:, i, :], 20)
    if i == 0:
        axs_sig1_v_cov[3, i].set_ylabel('joint_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[3, i].plot(sig_1, norm_cov, 'o')
    axs_sig1_v_cov[3, i].set_yticklabels([])
    axs_sig1_v_cov[3, i].set_xticklabels([])
    axs_sig1_v_cov[3, i].set_xticks([])
    axs_sig1_v_cov[3, i].set_yticks([])

plt.savefig('contours/sig1_v_cov.png', dpi=300)
plt.close()

fig_sig1_v_cov, axs_sig1_v_cov = plt.subplots(4, m)
for i in range(m):
    sig_plt = sig2_mesh[i, :, :]
    cov_plt = normalized_cov[i, :, :]
    axs_sig1_v_cov[0, i].contour(sig_plt, cov_plt, pca_reg_inner_vals[i, :, :], 20)
    if i == 2:
        axs_sig1_v_cov[0, i].set_title(f'sig1 = {Sig1_linspace[i]}')
    if i == 0:
        axs_sig1_v_cov[0, i].set_ylabel('pca_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[0, i].plot(sig_2, norm_cov, 'o')
    axs_sig1_v_cov[0, i].set_yticklabels([])
    axs_sig1_v_cov[0, i].set_xticklabels([])
    axs_sig1_v_cov[0, i].set_xticks([])
    axs_sig1_v_cov[0, i].set_yticks([])

    axs_sig1_v_cov[1, i].contour(sig_plt, cov_plt, pca_theory_vals[i, :, :], 20)
    if i == 0:
        axs_sig1_v_cov[1, i].set_ylabel('pca_reg_theory')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[1, i].plot(sig_2, norm_cov, 'o')
    axs_sig1_v_cov[1, i].set_yticklabels([])
    axs_sig1_v_cov[1, i].set_xticklabels([])
    axs_sig1_v_cov[1, i].set_xticks([])
    axs_sig1_v_cov[1, i].set_yticks([])

    axs_sig1_v_cov[2, i].contour(sig_plt, cov_plt, l1_vals[i, :, :], 20)
    if i == 0:
        axs_sig1_v_cov[2, i].set_ylabel('l1_std')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[2, i].plot(sig_2, norm_cov, 'o')
    axs_sig1_v_cov[2, i].set_yticklabels([])
    axs_sig1_v_cov[2, i].set_xticklabels([])
    axs_sig1_v_cov[2, i].set_xticks([])
    axs_sig1_v_cov[2, i].set_yticks([])

    axs_sig1_v_cov[3, i].contour(sig_plt, cov_plt, joint_reg[i, :, :], 20)
    if i == 0:
        axs_sig1_v_cov[3, i].set_ylabel('joint_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[3, i].plot(sig_2, norm_cov, 'o')
    axs_sig1_v_cov[3, i].set_yticklabels([])
    axs_sig1_v_cov[3, i].set_xticklabels([])
    axs_sig1_v_cov[3, i].set_xticks([])
    axs_sig1_v_cov[3, i].set_yticks([])

plt.savefig('contours/sig2_v_cov.png', dpi=300)
plt.close()

fig_sig1_v_cov, axs_sig1_v_cov = plt.subplots(4, m)
for i in range(m):
    sig_plt = sig1_mesh[:, :, i]
    cov_plt = sig2_mesh[:, :, i]
    axs_sig1_v_cov[0, i].contour(sig_plt, cov_plt, pca_reg_inner_vals[:, :, i], 20)
    if i == 2:
        axs_sig1_v_cov[0, i].set_title(f'sig1 = {Sig1_linspace[i]}')
    if i == 0:
        axs_sig1_v_cov[0, i].set_ylabel('pca_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[0, i].plot(sig_1, sig_2, 'o')
    axs_sig1_v_cov[0, i].set_yticklabels([])
    axs_sig1_v_cov[0, i].set_xticklabels([])
    axs_sig1_v_cov[0, i].set_xticks([])
    axs_sig1_v_cov[0, i].set_yticks([])

    axs_sig1_v_cov[1, i].contour(sig_plt, cov_plt, pca_theory_vals[:, :, i], 20)
    if i == 0:
        axs_sig1_v_cov[1, i].set_ylabel('pca_reg_theory')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[1, i].plot(sig_1, sig_2, 'o')
    axs_sig1_v_cov[1, i].set_yticklabels([])
    axs_sig1_v_cov[1, i].set_xticklabels([])
    axs_sig1_v_cov[1, i].set_xticks([])
    axs_sig1_v_cov[1, i].set_yticks([])

    axs_sig1_v_cov[2, i].contour(sig_plt, cov_plt, l1_vals[:, :, i], 20)
    if i == 0:
        axs_sig1_v_cov[2, i].set_ylabel('l1_std_thry')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[2, i].plot(sig_1, sig_2, 'o')
    axs_sig1_v_cov[2, i].set_yticklabels([])
    axs_sig1_v_cov[2, i].set_xticklabels([])
    axs_sig1_v_cov[2, i].set_xticks([])
    axs_sig1_v_cov[2, i].set_yticks([])

    axs_sig1_v_cov[3, i].contour(sig_plt, cov_plt, joint_reg[:, :, i], 20)
    if i == 0:
        axs_sig1_v_cov[3, i].set_ylabel('joint_reg')
    # axs_sig1_v_cov[0, i].set_ylabel('cross cov')
    axs_sig1_v_cov[3, i].plot(sig_1, sig_2, 'o')
    axs_sig1_v_cov[3, i].set_yticklabels([])
    axs_sig1_v_cov[3, i].set_xticklabels([])
    axs_sig1_v_cov[3, i].set_xticks([])
    axs_sig1_v_cov[3, i].set_yticks([])

plt.savefig('contours/sig1_v_sig2.png', dpi=300)
plt.close()
# plt.contour(sig1_mesh, sig2_mesh, pca_reg_inner_vals, 30)
# plt.colorbar()
# plt.title('pca_reg')
# plt.xlabel('sig_1, sig_2')
# plt.ylabel('cross cov')
# plt.plot(sig_1, norm_cov, 'o')
# plt.savefig(f'contours/pca_reg_sig_v_cov={cov}.png')
#
# plt.figure()
# plt.contour(sig1_mesh, sig2_mesh, pca_theory_vals, 30)
# plt.colorbar()
# plt.title('pca_reg')
# plt.xlabel('sig_1, sig_2')
# plt.ylabel('cross cov')
# plt.plot(sig_1, norm_cov, 'o')
# plt.savefig(f'contours/pca_reg_theory_sig_v_cov={cov}.png')
#
# plt.figure()
# plt.contour(sig1_mesh, sig2_mesh, l1_vals, 30)
# plt.colorbar()
# plt.title('l1_std_p')
# plt.xlabel('sig_1, sig_2')
# plt.ylabel('cross cov')
# plt.plot(sig_1, norm_cov, 'o')
# plt.savefig(f'contours/l1_std_p_sig_v_cov={cov}.png')
#
# plt.figure()
# plt.contour(sig1_mesh, sig2_mesh, joint_reg, 30)
# plt.colorbar()
# plt.title('l1_std_p + pca_reg')
# plt.xlabel('sig_1, sig_2')
# plt.ylabel('cross cov')
# plt.plot(sig_1, norm_cov, 'o')
# plt.savefig(f'contours/joint_reg_sig_v_cov={cov}.png')