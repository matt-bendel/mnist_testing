import torch

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F

from torchmetrics.functional import peak_signal_noise_ratio

from PIL import Image
from architectures.generator import Generator, Autoencoder
from architectures.unet import Unet


class MSENET(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.mean_net = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
        )

        self.resolution = self.args.im_size
        self.automatic_optimization = False
        self.val_outputs = []

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

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        opt1 = self.optimizers()

        x_hat = self.mean_net(y)
        # x_hat = self.readd_measures(x_hat, y)
        mu_loss = F.mse_loss(x_hat, x)

        opt1.zero_grad()
        self.manual_backward(mu_loss)
        opt1.step()

        self.log('mu_loss', mu_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081
        fig_count = 0

        x_hat = self.mean_net(y) * 0.3081 + 0.1307

        x = x * 0.3081 + 0.1307
        y = y * 0.3081 + 0.1307
        x_hat = self.readd_measures(x_hat, y)
        psnr_val = peak_signal_noise_ratio(x_hat, x)

        self.log('psnr_val_step', psnr_val, on_step=True, on_epoch=False, prog_bar=True)

        ############################################

        # Plot GT, Mean, 4 PCs
        if batch_idx <= 2:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                images = []

                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = x_hat[0, 0, :, :].cpu().numpy()

                plt.figure()
                plt.imshow(x_hat_np, cmap='gray')
                plt.savefig(f'test_recon_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(np.abs(x_np - x_hat_np), cmap='jet')
                plt.savefig(f'test_error_{batch_idx}.png')
                plt.close()

                y_np = y[0, 0, :, :].cpu().numpy()
                # images.append(Image.fromarray(np.uint8(x_np * 255), 'L'))
                # images.append(Image.fromarray(np.uint8(y_np * 255), 'L'))
                # images.append(Image.fromarray(np.uint8(x_hat_np * 255), 'L'))
                #
                # self.logger.log_image(
                #     key=f"epoch_{batch_idx}_{self.current_epoch}_img",
                #     images=images,
                #     caption=["x", "y", "x_hat"]
                # )

            self.trainer.strategy.barrier()

        self.val_outputs.append({'psnr_val': psnr_val})

        return {'psnr_val': psnr_val}

    def on_validation_epoch_end(self):
        psnr_val = torch.stack([x['psnr_val'] for x in self.val_outputs]).mean().mean()
        self.log('psnr_val', psnr_val)
        self.val_outputs = []

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        sch.step(self.trainer.callback_metrics["psnr_val"])

    def configure_optimizers(self):
        opt_mean = torch.optim.Adam(self.mean_net.parameters(), lr=self.args.lr,
                                    betas=(self.args.beta_1, self.args.beta_2))
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_mean,
            mode='max',
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        )
        return {'optimizer': opt_mean, 'lr_scheduler': reduce_lr_on_plateau}
