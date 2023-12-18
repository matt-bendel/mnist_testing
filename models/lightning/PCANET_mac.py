import torch

import pytorch_lightning as pl
import numpy as np
from torch.nn import functional as F

from matplotlib import cm
import matplotlib.pyplot as plt

from PIL import Image
from architectures.unet import Unet
from models.lightning.MSENET import MSENET

class PCANET(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.lamda_1 = 0.2
        self.lamda_2 = 0.1

        self.pca_net = Unet(
            in_chans=self.in_chans * 2,
            out_chans=self.out_chans * self.args.K,
        )

        self.mean_net = MSENET.load_from_checkpoint(self.args.checkpoint_dir + 'mse_model' + '/best-mse.ckpt').mean_net
        self.mean_net.eval()

        self.resolution = self.args.im_size
        self.automatic_optimization = False
        self.val_outputs = []

        self.save_hyperparameters()  # Save passed values

    def readd_measures(self, samples, measures):
        mask = torch.ones(samples.size(0), 1, 28, 28).to(samples.device)
        mask[:, :, 0:21, :] = 0
        samples = (1 - mask) * samples + measures

        return samples

    def forward(self, y, mean):
        input = torch.cat([y, mean], dim=1)
        directions = self.pca_net(input)
        return directions

    def gramm_schmidt(self, directions):
        principle_components = torch.zeros(directions.shape).to(directions.device)
        diff_vals = torch.zeros(directions.shape).to(directions.device)
        for k in range(directions.shape[1]):
            d = directions[:, k, :, :].clone()
            if k == 0:
                principle_components[:, 0, :, :] = d / torch.norm(d, p=2, dim=(1,2))[:, None, None]
            else:
                sum_val = torch.zeros(directions.shape[0], 28, 28).to(directions.device)
                for l in range(k):
                    detached_pc = principle_components[:, l, :, :].clone().detach()
                    inner_product = torch.sum(d * detached_pc, dim=(1,2))
                    sum_val = sum_val + inner_product[:, None, None] * detached_pc

                diff_vals[:, k, :, :] = d - sum_val
                diff_val_clone = diff_vals[:, k, :, :].clone()
                principle_components[:, k, :, :] = diff_val_clone / torch.norm(diff_val_clone, p=2, dim=(1,2))[:, None, None]

        return principle_components, diff_vals

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        opt2 = self.optimizers()

        with torch.no_grad():
            x_hat = self.mean_net(y) * 0.3081 + 0.1307
            y = y * 0.3081 + 0.1307
            x_hat = self.readd_measures(x_hat, y).detach()

        directions = self.forward(y, x_hat)
        x = x * 0.3081 + 0.1307
        principle_components, diff_vals = self.gramm_schmidt(directions)

        sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
        w_loss = torch.zeros(directions.shape[0]).to(directions.device)
        for k in range(directions.shape[1]):
            e_i_norm = torch.norm((x - x_hat)[:, 0, :, :], dim=(1,2))
            w_t_ei = torch.sum(principle_components[:, k, :, :] * (x - x_hat)[:, 0, :, :], dim=(1,2))
            w_t_ei_2 = w_t_ei ** 2 / e_i_norm ** 2

            sigma_loss += (torch.sum(diff_vals[:, k, :, :] * diff_vals[:, k, :, :], dim=(1,2)) - w_t_ei.clone().detach() ** 2) ** 2 / e_i_norm ** 4
            w_loss += w_t_ei_2

        sigma_loss = sigma_loss.sum()
        w_loss = - w_loss.sum()

        self.log('sigma_loss', sigma_loss, prog_bar=True)
        self.log('w_loss', w_loss, prog_bar=True)

        pca_loss = w_loss + sigma_loss

        opt2.zero_grad()
        self.manual_backward(pca_loss)
        opt2.step()

        self.log('pca_loss', pca_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        fig_count = 0
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        x_hat = self.mean_net(y) * 0.3081 + 0.1307
        x = x * 0.3081 + 0.1307
        y = y * 0.3081 + 0.1307
        x_hat = self.readd_measures(x_hat, y)

        directions = self.forward(y, x_hat)
        principle_components, diff_vals = self.gramm_schmidt(directions)

        sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
        w_loss = torch.zeros(directions.shape[0]).to(directions.device)
        for k in range(directions.shape[1]):
            e_i_norm = torch.norm((x - x_hat)[:, 0, :, :], dim=(1,2))
            w_t_ei = torch.sum(principle_components[:, k, :, :] * (x - x_hat)[:, 0, :, :], dim=(1, 2))
            w_t_ei_2 = w_t_ei ** 2 / e_i_norm ** 2

            sigma_loss += (torch.sum(diff_vals[:, k, :, :] * diff_vals[:, k, :, :],
                                     dim=(1, 2)) - w_t_ei.clone().detach() ** 2) ** 2 / e_i_norm ** 4
            w_loss += w_t_ei_2

        sigma_loss = sigma_loss.sum()
        w_loss = - w_loss.sum()

        self.log('w_loss_val', w_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('sigma_loss_val', sigma_loss, on_step=True, on_epoch=False, prog_bar=True)

        ############################################

        # Plot GT, Mean, 4 PCs
        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                images = []

                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = x_hat[0, 0, :, :].cpu().numpy()
                y_np = y[0, 0, :, :].cpu().numpy()

                plt.figure()
                plt.imshow(x_hat_np, cmap='gray')
                plt.savefig(f'test_recon_pca_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(x_np, cmap='gray')
                plt.savefig(f'test_gt_pca_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(y_np, cmap='gray')
                plt.savefig(f'test_y_pca_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(np.abs(x_np - x_hat_np), cmap='jet')
                plt.savefig(f'test_error_pca_{batch_idx}.png')
                plt.close()

                for k in range(4):
                    pc_np = (principle_components[0, k, :, :] * (1 - mask[0, 0, :, :])).cpu().numpy()

                    plt.figure()
                    plt.imshow(pc_np, cmap='bwr')
                    plt.savefig(f'test_pc_pca_{batch_idx}_{k}.png')
                    plt.colorbar()
                    plt.close()

            self.trainer.strategy.barrier()

        self.val_outputs.append({'w_val': w_loss})

        return {'w_loss_val': w_loss, 'sigma_loss_val': sigma_loss}

    def on_validation_epoch_end(self):
        w_val = torch.stack([x['w_val'] for x in self.val_outputs]).mean().mean()
        self.log('w_val', w_val)
        self.val_outputs = []

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        sch.step(self.trainer.callback_metrics["w_val"])

    def configure_optimizers(self):
        opt_pca = torch.optim.Adam(self.pca_net.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_pca,
            mode='max',
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        )
        return {'optimizer': opt_pca, 'lr_scheduler': reduce_lr_on_plateau}
