import torch

import pytorch_lightning as pl
import numpy as np
from torch.nn import functional as F

# from matplotlib import cm

from PIL import Image
from architectures.generator import Generator

class PCANET(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.lamda_1 = 1
        self.lamda_2 = 1

        self.pca_net = Generator(
            in_chans=self.in_chans,
            out_chans=self.out_chans * self.args.K,
            batch_norm=False,
            instance_norm=True
        )

        self.mean_net = Generator(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            batch_norm=False,
            instance_norm=True
        )

        self.resolution = self.args.im_size

        self.save_hyperparameters()  # Save passed values

    def readd_measures(self, samples, measures):
        mask = torch.ones(samples.size(0), 1, 28, 28).to(samples.device)
        mask[:, :, 0:21, :] = 0
        samples = (1 - mask) * samples + mask * measures

        return samples

    def forward(self, y, mean):
        input = torch.cat([y, mean], dim=1)
        directions = self.pca_net(input)
        return directions

    def gramm_schmidt(self, directions):
        principle_components = torch.zeros(directions.shape).to(directions.device)
        diff_vals = torch.zeros(directions.shape).to(directions.device)
        for k in range(directions.shape[1]):
            if k == 0:
                principle_components[:, 0, :, :] = directions[:, 0, :, :] / torch.norm(directions[:, 0, :, :], p=2, dim=(1,2))[:, None, None]
            else:
                sum_val = torch.zeros(directions.shape[0], 28, 28).to(directions.device)
                for l in range(k):
                    detached_pc = principle_components[:, l, :, :].detach()
                    sum_val += torch.sum(directions[:, k, :, :] * detached_pc, dim=(1,2))[:, None, None] * detached_pc

                diff_vals[:, k, :, :] = directions[:, k, :, :] - sum_val
                principle_components[:, k, :, :] = diff_vals[:, k, :, :] / torch.norm(diff_val, p=2, dim=(1,2))[:, None, None]

        return principle_components, diff_vals

    def training_step(self, batch, batch_idx, optimizer_idx):
        torch.autograd.set_detect_anomaly(True)
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask

        # train Mean net
        if optimizer_idx == 0:
            x_hat = self.mean_net(y)
            x_hat = self.readd_measures(x_hat, x)
            mu_loss = F.mse_loss(x_hat, x)

            self.log('mu_loss', mu_loss, prog_bar=True)

            return mu_loss

        # train PCA net
        if optimizer_idx == 1:
            x_hat = self.mean_net(y)
            directions = self.forward(y, x_hat)
            principle_components, diff_vals = self.gramm_schmidt(directions)

            sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
            w_loss = torch.zeros(directions.shape[0]).to(directions.device)
            for k in range(directions.shape[1]):
                w_t_ei = torch.sum(principle_components[:, k, :, :] * (x - x_hat)[:, 0, :, :], dim=(1,2))
                w_t_ei_2 = w_t_ei ** 2

                sigma_loss += (torch.sum(diff_vals[:, k, :, :] * diff_vals[:, k, :, :], dim=(1,2)) - w_t_ei.detach() ** 2) ** 2
                w_loss += w_t_ei_2

            sigma_loss = self.lamda_1 * sigma_loss.sum()
            w_loss = - self.lamda_2 * w_loss.sum()

            self.log('sigma_loss', sigma_loss, prog_bar=True)
            self.log('w_loss', w_loss, prog_bar=True)

            pca_loss = w_loss + sigma_loss
            self.log('pca_loss', pca_loss, prog_bar=True)

            return pca_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        fig_count = 0

        x_hat = self.mean_net(y)
        directions = self.forward(y, x_hat)
        principle_components, diff_vals = self.gramm_schmidt(directions)

        sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
        w_loss = torch.zeros(directions.shape[0]).to(directions.device)
        for k in range(directions.shape[1]):
            w_t_ei = torch.sum(principle_components[:, k, :, :] * (x - x_hat)[:, 0, :, :], dim=(1, 2))
            w_t_ei_2 = w_t_ei ** 2

            sigma_loss += (torch.sum(diff_vals[:, k, :, :] * diff_vals[:, k, :, :],
                                     dim=(1, 2)) - w_t_ei.detach() ** 2) ** 2
            w_loss += w_t_ei_2

        sigma_loss = self.lamda_1 * sigma_loss.sum()
        w_loss = - self.lamda_2 * w_loss.sum()

        self.log('w_loss_val', w_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('sigma_loss_val', sigma_loss, on_step=True, on_epoch=False, prog_bar=True)

        ############################################

        # Plot GT, Mean, 4 PCs
        # if batch_idx == 0:
        #     if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
        #         images = []
        #
        #         x_np = x[0, 0, :, :].cpu().numpy()
        #         x_hat_np = x_hat[0, 0, :, :].cpu().numpy()
        #         images.append(Image.fromarray(np.uint8(x_np * 255), 'L'))
        #         images.append(Image.fromarray(np.uint8(x_hat_np * 255), 'L'))
        #
        #         for k in range(4):
        #             pc_np = principle_components[0, k, :, :].cpu().numpy()
        #             images.append(Image.fromarray(np.uint8(cm.bwr(pc_np*255))))
        #
        #         self.logger.log_image(
        #             key=f"epoch_{self.current_epoch}_img",
        #             images=images,
        #             caption=["x", "x_hat", "PC1", "PC2", "PC3", "PC4"]
        #         )
        #
        #     self.trainer.strategy.barrier()

        return {'w_loss_val': w_loss, 'sigma_loss_val': sigma_loss}

    def configure_optimizers(self):
        opt_mean = torch.optim.Adam(self.mean_net.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_pca = torch.optim.Adam(self.pca_net.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return [opt_mean, opt_pca], []
